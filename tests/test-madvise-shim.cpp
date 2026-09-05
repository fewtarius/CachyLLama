// SPDX-License-Identifier: GPL-3.0-or-later
//
// Platform-agnostic madvise shim test.
//
// Exercises llama_moe_residency_madvise() and llama_moe_residency_pagesize()
// on a page-aligned anonymous mapping. The shim's contract is:
//   - Returns 0 on success for valid advice values.
//   - Returns -1 with errno == EINVAL for unknown advice values.
//   - The pagesize wrapper returns the OS page size in bytes (positive,
//     power-of-2).
//
// On Linux the shim forwards to the real madvise() / getpagesize(); on
// Windows it routes through PrefetchVirtualMemory and VirtualUnlock. The
// behavior under test is the platform-agnostic contract; the per-platform
// physical effect (page-cache prefetch, working-set trim) is documented
// but not directly observable from this test.

#include "llama-moe-residency.h"

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif

#undef NDEBUG
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static size_t alloc_anon_pages(size_t n_pages) {
    const int ps = llama_moe_residency_pagesize();
    assert(ps > 0);
    // Power-of-2 sanity check.
    assert((ps & (ps - 1)) == 0);

    const size_t len = (size_t) ps * n_pages;
#if defined(_WIN32)
    void * p = VirtualAlloc(nullptr, len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) {
        std::fprintf(stderr, "VirtualAlloc failed: %lu\n", GetLastError());
        std::abort();
    }
    return (size_t) p;
#else
    void * p = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::abort();
    }
    // Touch each page so the kernel has something to act on.
    std::memset(p, 0, len);
    return (size_t) p;
#endif
}

static void free_anon_pages(size_t addr, size_t n_pages) {
    const size_t len = (size_t) llama_moe_residency_pagesize() * n_pages;
#if defined(_WIN32)
    VirtualFree((void *) addr, 0, MEM_RELEASE);
#else
    ::munmap((void *) addr, len);
#endif
}

static void test_pagesize_wrapper() {
    const int ps = llama_moe_residency_pagesize();
    assert(ps == 4096 || ps == 8192 || ps == 16384 || ps == 65536);
    std::printf("test_pagesize_wrapper: ps=%d OK\n", ps);
}

static void test_willneed_dontneed_cold() {
    const size_t n_pages = 4;
    const size_t addr    = alloc_anon_pages(n_pages);
    const size_t len     = n_pages * (size_t) llama_moe_residency_pagesize();

    // WILLNEED: best-effort hint, must return success on a valid mapping.
    int rc = llama_moe_residency_madvise((void *) addr, len, LLAMA_MOE_RESIDENCY_MADV_WILLNEED);
    assert(rc == 0);

    // COLD: non-destructive hint on file-backed maps, no-op on anonymous
    // mappings on Linux. The shim must still report success.
    rc = llama_moe_residency_madvise((void *) addr, len, LLAMA_MOE_RESIDENCY_MADV_COLD);
    assert(rc == 0);

    // DONTNEED: same contract.
    rc = llama_moe_residency_madvise((void *) addr, len, LLAMA_MOE_RESIDENCY_MADV_DONTNEED);
    assert(rc == 0);

    free_anon_pages(addr, n_pages);
    std::printf("test_willneed_dontneed_cold: OK\n");
}

static void test_invalid_advice() {
    const size_t n_pages = 1;
    const size_t addr    = alloc_anon_pages(n_pages);
    const size_t len     = (size_t) llama_moe_residency_pagesize();

    // 999 is not a defined advice value on either platform.
    errno  = 0;
    int rc = llama_moe_residency_madvise((void *) addr, len, 999);
    assert(rc == -1);
    assert(errno == EINVAL);

    free_anon_pages(addr, n_pages);
    std::printf("test_invalid_advice: OK\n");
}

static void test_aligned_subrange() {
    // Caller in production passes a page-aligned base (safe_madvise() does
    // the alignment). The shim itself does not align - we only verify that
    // an aligned call succeeds.
    const size_t n_pages = 2;
    const size_t addr    = alloc_anon_pages(n_pages);
    const int    ps      = llama_moe_residency_pagesize();

    int rc = llama_moe_residency_madvise((void *) addr, (size_t) ps, LLAMA_MOE_RESIDENCY_MADV_WILLNEED);
    assert(rc == 0);

    free_anon_pages(addr, n_pages);
    std::printf("test_aligned_subrange: OK\n");
}

int main() {
    test_pagesize_wrapper();
    test_willneed_dontneed_cold();
    test_invalid_advice();
    test_aligned_subrange();
    std::printf("all madvise-shim tests passed\n");
    return 0;
}
