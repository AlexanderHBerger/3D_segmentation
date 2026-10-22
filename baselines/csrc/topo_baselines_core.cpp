// topo_baselines_core -- the Morse-graph kernel of the DMT loss, the authors' released
// pipeline (four processes glued by files) as one in-process computation with identical
// output, pinned to the released binaries by tests/baselines/test_dmt.py.
//
//   DMT loss (Hu, Wang, Fuxin, Samaras, Chen, ICLR 2021), github.com/HuXiaoling/DMT_loss,
//      folder morse_DMT/: the Morse graph (1-stable manifolds of the persistence-pruned
//      saddles) of a likelihood map. The released pipeline is
//         write_dipha_file_3d_revise.py : values int(-f * 255) on an nx x ny x 1 lattice
//         dipha (graph-recon fork)       : persistence of the lower-star cubical filtration,
//                                          one line per EDGE: (a, b, pers), a dim-0 death
//                                          edge encoded -(pers) - 1, a dim-1 birth edge pers
//         load_persistence_diagram.m     : binary -> text
//         ComputeGraphReconstruction.cpp : vector field = dim-0 death edges with pers <= thr;
//                                          critical edges = pers > thr; for each endpoint the
//                                          path through the vector-field tree to the tree's
//                                          minimum of (value, z, y, x); the union is the graph
//      morse_graph() below computes the same edge classification, persistences and paths
//      in one pass: a union-find over the edges in DIPHA's filtration order (value, dim,
//      cell index) for the dim-0 pairs, the dual union-find in the reversed order for the
//      dim-1 pairs (planar cut/cycle duality, the exterior as one extra face), and the same
//      BFS path tracing. Coordinates follow the reference: x = row i, y = column j, lattice
//      index i + H*j (x fastest), cubical cell (tx, ty) index tx + (2H-1)*ty.
//
// The batch entry point releases the GIL and runs one std::thread per sample.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

static inline int32_t uf_find(std::vector<int32_t>& parent, int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// ====================================================================================
// DMT: Morse graph of a quantised likelihood
// ====================================================================================

// Per-edge record of the persistence pass, in DIPHA cell-index order (the order the
// reference's edge file has after its sort is by (a, b), which only permutes the output).
struct MorseEdges {
    int64_t n = 0, m = 0;               // lattice: n rows (x, fast), m columns (y, slow)
    std::vector<int32_t> a;             // lower lattice index of the edge
    std::vector<uint8_t> is_x;          // 1: (i,j)-(i+1,j)  0: (i,j)-(i,j+1)
    std::vector<int32_t> pers;          // persistence in value units
    std::vector<uint8_t> type;          // 0: dim-0 death edge (negative), 1: dim-1 birth edge
    int64_t inconsistent = 0;           // primal/dual disagreements (must be 0)
};

// vals_lat: values in lattice order (index i + n*j), any int32 range.
static MorseEdges morse_edges(const std::vector<int32_t>& vals, int64_t n, int64_t m) {
    if (n < 2 || m < 2) throw std::invalid_argument("morse_graph needs at least 2 x 2 pixels");
    MorseEdges E;
    E.n = n;
    E.m = m;
    const int64_t N = n * m;
    const int64_t nE = m * (n - 1) + (m - 1) * n;
    E.a.resize(nE);
    E.is_x.resize(nE);
    E.pers.assign(nE, 0);
    E.type.assign(nE, 0);

    int32_t vmin = std::numeric_limits<int32_t>::max(), vmax = std::numeric_limits<int32_t>::min();
    for (int64_t p = 0; p < N; ++p) {
        vmin = std::min(vmin, vals[p]);
        vmax = std::max(vmax, vals[p]);
    }
    const int64_t nb = int64_t(vmax) - int64_t(vmin) + 1;

    // ---- edges in cell-index order: per column j, the x-edges (ty = 2j) then the y-edges
    // (ty = 2j+1); value = max of the two vertices
    std::vector<int32_t> eval(nE);
    {
        int64_t e = 0;
        for (int64_t j = 0; j < m; ++j) {
            for (int64_t i = 0; i < n - 1; ++i, ++e) {
                const int32_t p = int32_t(i + n * j);
                E.a[e] = p;
                E.is_x[e] = 1;
                eval[e] = std::max(vals[p], vals[p + 1]);
            }
            if (j < m - 1) {
                for (int64_t i = 0; i < n; ++i, ++e) {
                    const int32_t p = int32_t(i + n * j);
                    E.a[e] = p;
                    E.is_x[e] = 0;
                    eval[e] = std::max(vals[p], vals[p + n]);
                }
            }
        }
    }
    // ---- counting sort by value, stable in cell index: order[] ascending = DIPHA's
    // (value, dim=1, index) order restricted to edges
    std::vector<int64_t> bstart(nb + 1, 0);
    for (int64_t e = 0; e < nE; ++e) bstart[eval[e] - vmin + 1]++;
    for (int64_t b = 0; b < nb; ++b) bstart[b + 1] += bstart[b];
    std::vector<int32_t> order(nE);
    {
        std::vector<int64_t> fill(bstart.begin(), bstart.end() - 1);
        for (int64_t e = 0; e < nE; ++e) order[fill[eval[e] - vmin]++] = int32_t(e);
    }

    // ---- dim 0: union-find over the edges in ascending order; a component's birth is its
    // vertex of smallest (value, lattice index); the younger component dies at the edge
    {
        std::vector<int32_t> parent(N), birth(N);
        for (int64_t p = 0; p < N; ++p) { parent[p] = int32_t(p); birth[p] = int32_t(p); }
        auto younger = [&](int32_t u, int32_t v) {   // true if birth u is younger than birth v
            return vals[u] > vals[v] || (vals[u] == vals[v] && u > v);
        };
        for (int64_t k = 0; k < nE; ++k) {
            const int32_t e = order[k];
            const int32_t p = E.a[e];
            const int32_t q = E.is_x[e] ? p + 1 : int32_t(p + n);
            int32_t rp = uf_find(parent, p), rq = uf_find(parent, q);
            if (rp == rq) { E.type[e] = 1; continue; }
            int32_t young = rp, old = rq;
            if (younger(birth[rq], birth[rp])) { young = rq; old = rp; }
            E.pers[e] = eval[e] - vals[birth[young]];
            E.type[e] = 0;
            parent[young] = old;
        }
    }
    // ---- dim 1 through the dual: squares (+ the exterior) as nodes, edges inserted in
    // DESCENDING (value, index) order; a square-born component dying at edge e is the loop
    // e creates in the primal, killed by that square
    {
        const int64_t S = (n - 1) * (m - 1);
        const int32_t EXT = int32_t(S);
        std::vector<int32_t> sval(S);
        for (int64_t j = 0; j < m - 1; ++j)
            for (int64_t i = 0; i < n - 1; ++i) {
                const int64_t p = i + n * j;
                sval[i + (n - 1) * j] = std::max(std::max(vals[p], vals[p + 1]),
                                                 std::max(vals[p + n], vals[p + n + 1]));
            }
        std::vector<int32_t> parent(S + 1), birth(S + 1);
        for (int64_t s = 0; s <= S; ++s) { parent[s] = int32_t(s); birth[s] = int32_t(s); }
        auto coface = [&](int32_t e, int32_t& c0, int32_t& c1) {
            const int32_t p = E.a[e];
            const int64_t i = p % n, j = p / n;
            if (E.is_x[e]) {   // squares (i, j-1) and (i, j)
                c0 = (j >= 1) ? int32_t(i + (n - 1) * (j - 1)) : EXT;
                c1 = (j <= m - 2) ? int32_t(i + (n - 1) * j) : EXT;
            } else {           // squares (i-1, j) and (i, j)
                c0 = (i >= 1) ? int32_t((i - 1) + (n - 1) * j) : EXT;
                c1 = (i <= n - 2) ? int32_t(i + (n - 1) * j) : EXT;
            }
        };
        // elder in the reversed filtration = larger (value, id); the exterior is eldest
        auto elder = [&](int32_t s, int32_t t) {   // true if birth s is elder than birth t
            if (s == EXT) return true;
            if (t == EXT) return false;
            return sval[s] > sval[t] || (sval[s] == sval[t] && s > t);
        };
        for (int64_t k = nE - 1; k >= 0; --k) {
            const int32_t e = order[k];
            int32_t c0, c1;
            coface(e, c0, c1);
            int32_t r0 = uf_find(parent, c0), r1 = uf_find(parent, c1);
            if (r0 == r1) {
                if (E.type[e] != 0) E.inconsistent++;
                continue;
            }
            if (E.type[e] != 1) E.inconsistent++;
            int32_t old = r0, young = r1;
            if (elder(birth[r1], birth[r0])) { old = r1; young = r0; }
            E.pers[e] = sval[birth[young]] - eval[e];
            parent[young] = old;
        }
    }
    return E;
}

// The graph reconstruction of ComputeGraphReconstruction.cpp on the classified edges.
// Writes the Morse-graph pixel mask (lattice order) and stats.
static void morse_graph_from_edges(const std::vector<int32_t>& vals, const MorseEdges& E,
                                   int32_t thr, std::vector<uint8_t>& mask, int64_t* stats) {
    const int64_t n = E.n, m = E.m, N = n * m, nE = int64_t(E.a.size());
    // vector field: dim-0 death edges with pers <= thr; critical: pers > thr (either type)
    std::vector<int32_t> deg(N + 1, 0);
    int64_t n_vf = 0, n_crit = 0, n_crit_neg = 0, n_neg = 0;
    for (int64_t e = 0; e < nE; ++e) {
        if (E.type[e] == 0) n_neg++;
        if (E.pers[e] > thr) { n_crit++; if (E.type[e] == 0) n_crit_neg++; continue; }
        if (E.type[e] == 0) {
            const int32_t p = E.a[e], q = E.is_x[e] ? p + 1 : int32_t(p + n);
            deg[p]++; deg[q]++; n_vf++;
        }
    }
    std::vector<int64_t> off(N + 1, 0);
    for (int64_t p = 0; p < N; ++p) off[p + 1] = off[p] + deg[p];
    std::vector<int32_t> adj(off[N]);
    {
        std::vector<int64_t> fill(off.begin(), off.end() - 1);
        for (int64_t e = 0; e < nE; ++e) {
            if (E.type[e] != 0 || E.pers[e] > thr) continue;
            const int32_t p = E.a[e], q = E.is_x[e] ? p + 1 : int32_t(p + n);
            adj[fill[p]++] = q;
            adj[fill[q]++] = p;
        }
    }
    mask.assign(N, 0);
    std::vector<int32_t> min_computed(N, -1), pre(N, -1);
    std::vector<uint8_t> visited(N, 0);
    std::vector<int32_t> queue;
    queue.reserve(1 << 16);
    int64_t n_components = 0;

    // BFS from `start` over the vector field; returns the component's minimum of
    // (value, y = j, x = i) and leaves pre[] as the BFS tree from `start`.
    auto bfs = [&](int32_t start, std::vector<int32_t>& comp) -> int32_t {
        comp.clear();
        queue.clear();
        queue.push_back(start);
        visited[start] = 1;
        pre[start] = -1;
        int32_t best = start;
        size_t head = 0;
        while (head < queue.size()) {
            const int32_t v = queue[head++];
            comp.push_back(v);
            // (value, j, i) lexicographic minimum == (value, lattice index) minimum
            if (vals[v] < vals[best] || (vals[v] == vals[best] && v < best)) best = v;
            for (int64_t k = off[v]; k < off[v + 1]; ++k) {
                const int32_t w = adj[k];
                if (!visited[w]) { visited[w] = 1; pre[w] = v; queue.push_back(w); }
            }
        }
        for (int32_t v : comp) visited[v] = 0;
        return best;
    };
    std::vector<int32_t> comp;
    for (int64_t e = 0; e < nE; ++e) {
        if (E.pers[e] <= thr) continue;
        const int32_t p = E.a[e], q = E.is_x[e] ? p + 1 : int32_t(p + n);
        mask[p] = 1;
        mask[q] = 1;
        for (int32_t v : {p, q}) {
            if (min_computed[v] == -1) {
                const int32_t root = bfs(v, comp);
                for (int32_t u : comp) min_computed[u] = root;
                bfs(root, comp);      // pre[] = tree from the minimum
                n_components++;
            }
            for (int32_t u = v; u != -1; u = pre[u]) mask[u] = 1;
        }
    }
    int64_t n_mask = 0;
    for (int64_t p = 0; p < N; ++p) n_mask += mask[p];
    stats[0] = n_neg;                 // dim-0 death edges (= N - 1)
    stats[1] = nE - n_neg;            // dim-1 birth edges (= squares)
    stats[2] = n_crit;                // critical edges (pers > thr)
    stats[3] = n_crit_neg;            // ... of which dim-0 saddles
    stats[4] = n_vf;                  // vector-field edges
    stats[5] = n_components;          // vector-field trees traversed
    stats[6] = n_mask;                // Morse-graph pixels
    stats[7] = E.inconsistent;        // primal/dual disagreements (0)
}

// row-major (H, W) image -> lattice order (index i + H*j)
template <typename T>
static void to_lattice(const T* img, int64_t H, int64_t W, std::vector<T>& out) {
    out.resize(H * W);
    for (int64_t i = 0; i < H; ++i)
        for (int64_t j = 0; j < W; ++j) out[i + H * j] = img[i * W + j];
}

static void morse_graph_one(const int32_t* img, int64_t H, int64_t W, int32_t thr,
                            uint8_t* mask_out, int64_t* stats) {
    std::vector<int32_t> vals;
    to_lattice(img, H, W, vals);
    MorseEdges E = morse_edges(vals, H, W);
    std::vector<uint8_t> mask;
    morse_graph_from_edges(vals, E, thr, mask, stats);
    for (int64_t i = 0; i < H; ++i)
        for (int64_t j = 0; j < W; ++j) mask_out[i * W + j] = mask[i + H * j];
}


// ====================================================================================
// Homotopy warping, 3D (Hu, NeurIPS 2022): the two kernels of the released 3D branch that
// dominate its cost, with identical output. Everything else of the released per-sample body
// (the FP/FN masks, the products with the distance transforms, numpy's own argsort for the
// flipping order, the critical mask) stays in Python in baselines/warping.py, so the tie order is
// numpy's exactly as released.
//   edt3d           exact Euclidean distance to the nearest zero voxel (sqrt of the exact
//                   squared distance in double = scipy.ndimage.distance_transform_edt's
//                   values, bit for bit); INF where the volume has no zero voxel.
//   warp3d          decide_simple_point_3D applied in the given order: the 3x3x3 patch
//                   around the voxel is taken exactly as the released slice
//                   gt[x-1:x+2, y-1:y+2, z-1:z+2] (truncated at the far border, EMPTY at
//                   index 0 and then skipped by the released shape guard); the foreground
//                   and the background of that patch are each counted with 26-connectivity
//                   -- what cc3d.connected_components(patch, 6) computes, the positional 6
//                   being max_labels (a hint that does not raise), the connectivity its
//                   default 26 -- and the voxel flips iff n_fg * n_bg == 1.
// ====================================================================================

static const double DT_INF = 1e20;

// Felzenszwalb & Huttenlocher 1-D squared distance transform along a strided line.
static void dt1d(const double* f, int64_t len, int64_t stride, double* d, int32_t* v,
                 double* z) {
    auto F = [&](int64_t q) { return f[q * stride]; };
    auto isect = [&](int64_t q, int64_t r) {
        return ((F(q) + double(q) * double(q)) - (F(r) + double(r) * double(r))) /
               (2.0 * (double(q) - double(r)));
    };
    int k = 0;
    v[0] = 0;
    z[0] = -DT_INF;
    z[1] = +DT_INF;
    for (int64_t q = 1; q < len; ++q) {
        double s = isect(q, v[k]);
        while (s <= z[k]) {
            --k;
            s = isect(q, v[k]);
        }
        ++k;
        v[k] = int32_t(q);
        z[k] = s;
        z[k + 1] = +DT_INF;
    }
    k = 0;
    for (int64_t q = 0; q < len; ++q) {
        while (z[k + 1] < double(q)) ++k;
        const double dq = double(q) - double(v[k]);
        d[q * stride] = dq * dq + F(v[k]);
    }
}

// out = sqrt(exact squared distance to the nearest zero voxel), (D, H, W) row-major.
static void edt3d(const uint8_t* img, int64_t D, int64_t H, int64_t W, double* out) {
    const int64_t N = D * H * W;
    std::vector<double> a(N), b(N);
    bool any_zero = false;
    for (int64_t p = 0; p < N; ++p) { a[p] = img[p] ? DT_INF : 0.0; any_zero |= (img[p] == 0); }
    if (!any_zero) {
        for (int64_t p = 0; p < N; ++p) out[p] = DT_INF;
        return;
    }
    const int64_t L = std::max(D, std::max(H, W)) + 1;
    std::vector<int32_t> v(L);
    std::vector<double> z(L + 1);
    // along W
    for (int64_t d = 0; d < D; ++d)
        for (int64_t h = 0; h < H; ++h)
            dt1d(a.data() + (d * H + h) * W, W, 1, b.data() + (d * H + h) * W, v.data(), z.data());
    // along H
    for (int64_t d = 0; d < D; ++d)
        for (int64_t w = 0; w < W; ++w)
            dt1d(b.data() + d * H * W + w, H, W, a.data() + d * H * W + w, v.data(), z.data());
    // along D
    for (int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w)
            dt1d(a.data() + h * W + w, D, H * W, b.data() + h * W + w, v.data(), z.data());
    for (int64_t p = 0; p < N; ++p) out[p] = (b[p] >= DT_INF / 2) ? DT_INF : std::sqrt(b[p]);
}

// 26-connected component count of the voxels of `patch` (up to 27, row-major over the
// sub-box dims) equal to `value`.
static int count_components(const uint8_t* patch, int nd, int nh, int nw, uint8_t value) {
    const int n = nd * nh * nw;
    uint8_t seen[27] = {0};
    int stack[27];
    int count = 0;
    for (int s = 0; s < n; ++s) {
        if (seen[s] || patch[s] != value) continue;
        ++count;
        int top = 0;
        stack[top++] = s;
        seen[s] = 1;
        while (top) {
            const int c = stack[--top];
            const int cd = c / (nh * nw), ch = (c / nw) % nh, cw = c % nw;
            for (int dd = -1; dd <= 1; ++dd)
                for (int dh = -1; dh <= 1; ++dh)
                    for (int dw = -1; dw <= 1; ++dw) {
                        const int od = cd + dd, oh = ch + dh, ow = cw + dw;
                        if (od < 0 || od >= nd || oh < 0 || oh >= nh || ow < 0 || ow >= nw) continue;
                        const int o = (od * nh + oh) * nw + ow;
                        if (!seen[o] && patch[o] == value) { seen[o] = 1; stack[top++] = o; }
                    }
        }
    }
    return count;
}

// Flip the voxels of `order` (flat row-major indices, in visiting order) that are simple at
// the moment they are visited. Returns the number of flips.
static int64_t warp3d(uint8_t* mask, int64_t D, int64_t H, int64_t W, const int64_t* order,
                      int64_t n_order) {
    int64_t flips = 0;
    uint8_t patch[27];
    for (int64_t i = 0; i < n_order; ++i) {
        const int64_t p = order[i];
        const int64_t x = p / (H * W), y = (p / W) % H, zc = p % W;
        // the released slice gt[x-1:x+2, ...]: at index 0 the start wraps to the end and
        // the slice is empty -> the shape guard skips the voxel; at the far border it is
        // truncated to the in-bounds part.
        if (x == 0 || y == 0 || zc == 0) continue;
        const int64_t x0 = x - 1, x1 = std::min(x + 2, D), y0 = y - 1, y1 = std::min(y + 2, H),
                      z0 = zc - 1, z1 = std::min(zc + 2, W);
        const int nd = int(x1 - x0), nh = int(y1 - y0), nw = int(z1 - z0);
        int k = 0;
        for (int64_t a = x0; a < x1; ++a)
            for (int64_t b = y0; b < y1; ++b)
                for (int64_t c = z0; c < z1; ++c) patch[k++] = mask[(a * H + b) * W + c];
        const int nf = count_components(patch, nd, nh, nw, 1);
        const int nb = count_components(patch, nd, nh, nw, 0);
        if (nf * nb == 1) { mask[p] ^= 1; ++flips; }
    }
    return flips;
}

// ---------------------------------------------------------------- threading helper
template <typename F>
static void run_threads(int64_t B, int n_threads, F&& work) {
    n_threads = std::max(1, std::min<int>(n_threads, int(B)));
    std::vector<std::thread> pool;
    std::vector<std::string> errors(n_threads);
    for (int t = 0; t < n_threads; ++t)
        pool.emplace_back([&, t]() {
            try {
                for (int64_t b = t; b < B; b += n_threads) work(b);
            } catch (const std::exception& ex) {
                errors[t] = ex.what();
            }
        });
    for (auto& th : pool) th.join();
    for (const auto& e : errors)
        if (!e.empty()) throw std::runtime_error(e);
}

}  // namespace

// ====================================================================================
// bindings
// ====================================================================================

static py::tuple py_morse_graph_batch(py::array_t<int32_t, py::array::c_style | py::array::forcecast> vals,
                                      int32_t threshold, int n_threads) {
    if (vals.ndim() != 3) throw std::invalid_argument("vals must be (B, H, W) int32");
    const int64_t B = vals.shape(0), H = vals.shape(1), W = vals.shape(2);
    py::array_t<uint8_t> masks({B, H, W});
    py::array_t<int64_t> stats({B, int64_t(8)});
    const int32_t* vp = vals.data();
    uint8_t* mp = masks.mutable_data();
    int64_t* sp = stats.mutable_data();
    {
        py::gil_scoped_release nogil;
        run_threads(B, n_threads, [&](int64_t b) {
            morse_graph_one(vp + b * H * W, H, W, threshold, mp + b * H * W, sp + b * 8);
        });
    }
    return py::make_tuple(masks, stats);
}

// Fine-grained view for the parity test: every edge as (a, b, pers, type) in lattice
// indices, in DIPHA cell-index order.
static py::tuple py_morse_edges(py::array_t<int32_t, py::array::c_style | py::array::forcecast> vals) {
    if (vals.ndim() != 2) throw std::invalid_argument("vals must be (H, W) int32");
    const int64_t H = vals.shape(0), W = vals.shape(1);
    std::vector<int32_t> lat;
    to_lattice(vals.data(), H, W, lat);
    MorseEdges E = morse_edges(lat, H, W);
    const int64_t nE = int64_t(E.a.size());
    py::array_t<int32_t> a(nE), b(nE), pers(nE);
    py::array_t<uint8_t> type(nE);
    for (int64_t e = 0; e < nE; ++e) {
        a.mutable_data()[e] = E.a[e];
        b.mutable_data()[e] = E.is_x[e] ? E.a[e] + 1 : int32_t(E.a[e] + H);
        pers.mutable_data()[e] = E.pers[e];
        type.mutable_data()[e] = E.type[e];
    }
    return py::make_tuple(a, b, pers, type, E.inconsistent);
}


static py::array_t<double> py_edt3d(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
    if (img.ndim() != 3) throw std::invalid_argument("img must be (D, H, W) uint8");
    const int64_t D = img.shape(0), H = img.shape(1), W = img.shape(2);
    py::array_t<double> out({D, H, W});
    {
        py::gil_scoped_release nogil;
        edt3d(img.data(), D, H, W, out.mutable_data());
    }
    return out;
}

static py::tuple py_warp3d(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
                           py::array_t<int64_t, py::array::c_style | py::array::forcecast> order) {
    if (mask.ndim() != 3) throw std::invalid_argument("mask must be (D, H, W) uint8");
    if (order.ndim() != 1) throw std::invalid_argument("order must be a 1-D int64 array of flat indices");
    const int64_t D = mask.shape(0), H = mask.shape(1), W = mask.shape(2), n = order.shape(0);
    for (int64_t i = 0; i < n; ++i)
        if (order.data()[i] < 0 || order.data()[i] >= D * H * W) throw std::out_of_range("order index");
    py::array_t<uint8_t> out({D, H, W});
    std::memcpy(out.mutable_data(), mask.data(), D * H * W);
    int64_t flips;
    {
        py::gil_scoped_release nogil;
        flips = warp3d(out.mutable_data(), D, H, W, order.data(), n);
    }
    return py::make_tuple(out, flips);
}

PYBIND11_MODULE(topo_baselines_core, mod) {
    mod.doc() = "DMT (ICLR 2021) Morse-graph kernel and the 3D homotopy-warping (NeurIPS 2022) kernels, pinned to the released code";
    mod.def("morse_graph_batch", &py_morse_graph_batch, py::arg("vals"), py::arg("threshold"),
            py::arg("n_threads") = 8,
            "Morse-graph masks (B,H,W) uint8 + stats (B,8) int64 of quantised likelihoods (B,H,W) int32");
    mod.def("morse_edges", &py_morse_edges, py::arg("vals"),
            "(a, b, pers, type, n_inconsistent): every edge of the cubical complex of one (H,W) map");
    mod.def("edt3d", &py_edt3d, py::arg("img"),
            "exact Euclidean distance to the nearest zero voxel, (D,H,W) float64 = scipy's values");
    mod.def("warp3d", &py_warp3d, py::arg("mask"), py::arg("order"),
            "(warped mask, flips): decide_simple_point_3D of the released code in the given order");
}
