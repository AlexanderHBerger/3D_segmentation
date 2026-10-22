// dmt3d_core -- the Morse-graph kernel of the DMT loss in 3D: the authors' released pipeline
// (four processes glued by files) run on a VOLUME, as one in-process computation with
// identical output, pinned to the released binaries by tests/baselines/test_dmt.py.
//
//   DMT loss (Hu, Wang, Fuxin, Samaras, Chen, ICLR 2021), github.com/HuXiaoling/DMT_loss @
//   b50bc35, folder morse_DMT/ -- the route its README gives for "2D and 3D vessel data":
//      write_dipha_file_3d_revise.py : values int(-f * 255) on an nx x ny x nz lattice
//                                      (x fastest; the stack index is z)
//      dipha (graph-recon fork)       : persistence of the lower-star cubical filtration in
//                                      DIPHA's total order (value, dim, cell index), one line
//                                      per EDGE of a dim-0 or dim-1 pair: (a, b, pers), a dim-0
//                                      death edge encoded -(pers) - 1, a dim-1 birth edge pers
//      load_persistence_diagram.m     : binary -> text
//      ComputeGraphReconstruction.cpp : vector field = dim-0 death edges with pers <= thr;
//                                      critical edges = pers > thr; for each endpoint the path
//                                      through its vector-field tree to the tree's minimum of
//                                      (value, z, y, x); the union is the graph
//
// The 2D kernel (topo_baselines_core.cpp) gets its dim-1 pairs from the planar dual; a volume
// has none, so here the three dimensions are computed as follows, all in DIPHA's total order,
// which fixes the persistence pairing uniquely (the pairing lemma), so any correct reduction
// in that order reproduces the reference's pairs exactly:
//   dim 0  union-find over the edges in ascending order, elder rule (as in 2D);
//   dim 2  union-find over the dual graph -- cubes plus the exterior as nodes, squares as arcs,
//          inserted in DESCENDING order (the 2D code's dim-1 construction one dimension up);
//          a square joining two dual components is a dim-2 birth, every other square a dim-1
//          death;
//   dim 1  the standard column reduction of the dim-1-death squares only (clearing, from dim 2)
//          in ascending order, rows of dim-0-death edges dropped (compression: the youngest
//          edge of any cycle is a dim-1 birth, so no pivot ever sits on a dropped row); the
//          pivot edge of a square's reduced column is born at that edge and dies at that square.
// Cell indexing follows DIPHA's weighted_cubical_complex: cubical coordinates (tx, ty, tz) on
// the (2nx-1) x (2ny-1) x (2nz-1) grid, index tx + (2nx-1) * (ty + (2ny-1) * tz); a cell's value
// is the max over its vertices; an edge's boundary is (lower vertex, upper vertex); vertex
// (x, y, z) has lattice index x + nx * (y + ny * z) (compute_original_index).
//
// The batch entry point releases the GIL and runs the samples on a pool of std::threads.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

// Stable counting sort of `val` (values in [vmin, vmax]): order[rank] = id, ascending
// (value, id) -- DIPHA's order within one dimension when ids follow the cell index.
static void counting_order(const std::vector<int32_t>& val, int32_t vmin, int32_t vmax,
                           std::vector<int32_t>& order, std::vector<int32_t>* rank) {
    const int64_t n = int64_t(val.size());
    const int64_t nb = int64_t(vmax) - int64_t(vmin) + 1;
    std::vector<int64_t> start(nb + 1, 0);
    for (int64_t e = 0; e < n; ++e) start[val[e] - vmin + 1]++;
    for (int64_t b = 0; b < nb; ++b) start[b + 1] += start[b];
    order.resize(n);
    for (int64_t e = 0; e < n; ++e) order[start[val[e] - vmin]++] = int32_t(e);
    if (rank) {
        rank->resize(n);
        for (int64_t r = 0; r < n; ++r) (*rank)[order[r]] = int32_t(r);
    }
}

// Per-edge record in DIPHA cell-index order.
struct MorseEdges3D {
    int64_t nx = 0, ny = 0, nz = 0;
    std::vector<int32_t> a;       // lower vertex (lattice index)
    std::vector<uint8_t> dir;     // 0: +x, 1: +y, 2: +z
    std::vector<int32_t> pers;    // persistence in value units
    std::vector<uint8_t> type;    // 0: dim-0 death edge, 1: dim-1 birth edge
    int64_t inconsistent = 0;     // dual / reduction disagreements (must be 0)
    int64_t n_dim2 = 0;           // dim-2 pairs (squares born, cubes killing)
    int64_t n_additions = 0;      // column additions of the dim-1 reduction (cost diagnostic)
    int32_t other(int64_t e) const {
        const int64_t s = dir[e] == 0 ? 1 : (dir[e] == 1 ? nx : nx * ny);
        return int32_t(a[e] + s);
    }
};

// vals: values in lattice order (index x + nx*(y + ny*z)).
static MorseEdges3D morse_edges_3d(const std::vector<int32_t>& vals, int64_t nx, int64_t ny,
                                   int64_t nz) {
    if (nx < 2 || ny < 2 || nz < 1)
        throw std::invalid_argument("morse_graph_3d needs at least 2 x 2 x 1 voxels");
    const int64_t N = nx * ny * nz;
    if (3 * N >= int64_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("volume too large for int32 cell ids");
    const int64_t sy = nx, sz = nx * ny;
    MorseEdges3D E;
    E.nx = nx; E.ny = ny; E.nz = nz;

    int32_t vmin = std::numeric_limits<int32_t>::max(), vmax = std::numeric_limits<int32_t>::min();
    for (int64_t p = 0; p < N; ++p) { vmin = std::min(vmin, vals[p]); vmax = std::max(vmax, vals[p]); }

    // ---- edges in cell-index order. Layer tz = 2k: rows ty = 2j hold the x-edges (tx odd),
    // rows ty = 2j+1 the y-edges (tx even). Layer tz = 2k+1: rows ty = 2j hold the z-edges.
    std::vector<int32_t> eval;
    {
        const int64_t nE = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
        E.a.reserve(nE); E.dir.reserve(nE); eval.reserve(nE);
        for (int64_t k = 0; k < nz; ++k) {
            for (int64_t j = 0; j < ny; ++j) {
                for (int64_t i = 0; i + 1 < nx; ++i) {
                    const int32_t p = int32_t(i + sy * j + sz * k);
                    E.a.push_back(p); E.dir.push_back(0);
                    eval.push_back(std::max(vals[p], vals[p + 1]));
                }
                if (j + 1 < ny)
                    for (int64_t i = 0; i < nx; ++i) {
                        const int32_t p = int32_t(i + sy * j + sz * k);
                        E.a.push_back(p); E.dir.push_back(1);
                        eval.push_back(std::max(vals[p], vals[p + sy]));
                    }
            }
            if (k + 1 < nz)
                for (int64_t j = 0; j < ny; ++j)
                    for (int64_t i = 0; i < nx; ++i) {
                        const int32_t p = int32_t(i + sy * j + sz * k);
                        E.a.push_back(p); E.dir.push_back(2);
                        eval.push_back(std::max(vals[p], vals[p + sz]));
                    }
        }
    }
    const int64_t nE = int64_t(E.a.size());
    E.pers.assign(nE, 0);
    E.type.assign(nE, 0);
    std::vector<int32_t> eorder, erank;
    counting_order(eval, vmin, vmax, eorder, &erank);

    // edge id of the edge leaving vertex p in direction d (valid only if it exists)
    // per-layer offsets: layer k starts at k * (edges per tz-even layer + z-edges per layer)
    const int64_t n_xy_layer = (nx - 1) * ny + nx * (ny - 1);   // x- and y-edges in one z-slice
    const int64_t n_z_layer = nx * ny;                          // z-edges between two slices
    auto edge_id = [&](int64_t i, int64_t j, int64_t k, int d) -> int32_t {
        const int64_t base = k * (n_xy_layer + n_z_layer);
        if (d == 0) return int32_t(base + j * (2 * nx - 1) + i);
        if (d == 1) return int32_t(base + j * (2 * nx - 1) + (nx - 1) + i);
        return int32_t(base + n_xy_layer + j * nx + i);
    };

    // ---- dim 0: union-find over the edges in ascending order; a component's birth is its
    // vertex of smallest (value, lattice index); the younger component dies at the edge
    {
        std::vector<int32_t> parent(N), birth(N);
        for (int64_t p = 0; p < N; ++p) { parent[p] = int32_t(p); birth[p] = int32_t(p); }
        auto younger = [&](int32_t u, int32_t v) {
            return vals[u] > vals[v] || (vals[u] == vals[v] && u > v);
        };
        for (int64_t r = 0; r < nE; ++r) {
            const int32_t e = eorder[r];
            const int32_t p = E.a[e], q = E.other(e);
            const int32_t rp = uf_find(parent, p), rq = uf_find(parent, q);
            if (rp == rq) { E.type[e] = 1; continue; }
            int32_t young = rp, old = rq;
            if (younger(birth[rq], birth[rp])) { young = rq; old = rp; }
            E.pers[e] = eval[e] - vals[birth[young]];
            E.type[e] = 0;
            parent[young] = old;
        }
    }

    // ---- squares in cell-index order. Layer tz = 2k: rows ty = 2j+1 hold the xy-squares
    // (tx odd). Layer tz = 2k+1: rows ty = 2j hold the xz-squares (tx odd), rows ty = 2j+1 the
    // yz-squares (tx even). Each square: its four edges and its two cofaces (cube or exterior).
    std::vector<int32_t> sval;
    std::vector<int32_t> sedge;      // 4 edge ids per square
    std::vector<int32_t> scof;       // 2 coface ids per square (cube id or EXT)
    const int64_t nCx = nx - 1, nCy = ny - 1, nCz = nz - 1;
    const int64_t nC = nCx * nCy * std::max<int64_t>(nCz, 0);
    const int32_t EXT = int32_t(nC);
    auto cube_id = [&](int64_t i, int64_t j, int64_t k) -> int32_t {
        if (i < 0 || j < 0 || k < 0 || i >= nCx || j >= nCy || k >= nCz) return EXT;
        return int32_t(i + nCx * (j + nCy * k));
    };
    {
        const int64_t nS = nCx * nCy * nz + nCx * ny * nCz + nx * nCy * nCz;
        sval.reserve(nS); sedge.reserve(4 * nS); scof.reserve(2 * nS);
        auto vmax4 = [&](int64_t p, int64_t s1, int64_t s2) {
            return std::max(std::max(vals[p], vals[p + s1]), std::max(vals[p + s2], vals[p + s1 + s2]));
        };
        for (int64_t k = 0; k < nz; ++k) {
            for (int64_t j = 0; j + 1 < ny; ++j)
                for (int64_t i = 0; i + 1 < nx; ++i) {           // xy-square at (i, j, k)
                    const int64_t p = i + sy * j + sz * k;
                    sval.push_back(vmax4(p, 1, sy));
                    sedge.push_back(edge_id(i, j, k, 0));
                    sedge.push_back(edge_id(i, j + 1, k, 0));
                    sedge.push_back(edge_id(i, j, k, 1));
                    sedge.push_back(edge_id(i + 1, j, k, 1));
                    scof.push_back(cube_id(i, j, k - 1));
                    scof.push_back(cube_id(i, j, k));
                }
            if (k + 1 < nz)
                for (int64_t j = 0; j < ny; ++j) {
                    for (int64_t i = 0; i + 1 < nx; ++i) {       // xz-square at (i, j, k)
                        const int64_t p = i + sy * j + sz * k;
                        sval.push_back(vmax4(p, 1, sz));
                        sedge.push_back(edge_id(i, j, k, 0));
                        sedge.push_back(edge_id(i, j, k + 1, 0));
                        sedge.push_back(edge_id(i, j, k, 2));
                        sedge.push_back(edge_id(i + 1, j, k, 2));
                        scof.push_back(cube_id(i, j - 1, k));
                        scof.push_back(cube_id(i, j, k));
                    }
                    if (j + 1 < ny)
                        for (int64_t i = 0; i < nx; ++i) {       // yz-square at (i, j, k)
                            const int64_t p = i + sy * j + sz * k;
                            sval.push_back(vmax4(p, sy, sz));
                            sedge.push_back(edge_id(i, j, k, 1));
                            sedge.push_back(edge_id(i, j, k + 1, 1));
                            sedge.push_back(edge_id(i, j, k, 2));
                            sedge.push_back(edge_id(i, j + 1, k, 2));
                            scof.push_back(cube_id(i - 1, j, k));
                            scof.push_back(cube_id(i, j, k));
                        }
                }
        }
    }
    const int64_t nS = int64_t(sval.size());
    std::vector<int32_t> sorder;
    counting_order(sval, vmin, vmax, sorder, nullptr);

    // ---- dim 2: dual union-find, squares in DESCENDING (value, index); nodes = cubes + EXT,
    // elder = larger (value, cube id) (later in the forward order), the exterior eldest
    std::vector<uint8_t> s_neg(nS, 0);     // 1: the square kills a dim-1 class
    {
        std::vector<int32_t> cval(nC);
        for (int64_t k = 0; k < nCz; ++k)
            for (int64_t j = 0; j < nCy; ++j)
                for (int64_t i = 0; i < nCx; ++i) {
                    const int64_t p = i + sy * j + sz * k;
                    int32_t v = vals[p];
                    for (int64_t dz = 0; dz < 2; ++dz)
                        for (int64_t dy = 0; dy < 2; ++dy)
                            for (int64_t dx = 0; dx < 2; ++dx)
                                v = std::max(v, vals[p + dx + sy * dy + sz * dz]);
                    cval[i + nCx * (j + nCy * k)] = v;
                }
        std::vector<int32_t> parent(nC + 1), birth(nC + 1);
        for (int64_t c = 0; c <= nC; ++c) { parent[c] = int32_t(c); birth[c] = int32_t(c); }
        auto elder = [&](int32_t s, int32_t t) {
            if (s == EXT) return true;
            if (t == EXT) return false;
            return cval[s] > cval[t] || (cval[s] == cval[t] && s > t);
        };
        for (int64_t r = nS - 1; r >= 0; --r) {
            const int32_t s = sorder[r];
            const int32_t r0 = uf_find(parent, scof[2 * s]), r1 = uf_find(parent, scof[2 * s + 1]);
            if (r0 == r1) { s_neg[s] = 1; continue; }
            int32_t old = r0, young = r1;
            if (elder(birth[r1], birth[r0])) { old = r1; young = r0; }
            parent[young] = old;
            E.n_dim2++;
        }
    }

    // ---- dim 1: reduce the dim-1-death squares in ascending order; rows = dim-1-birth edges
    // by rank. Reduced columns live in one append-only pool.
    {
        std::vector<int32_t> owner(nE, -1);        // edge rank -> column slot in the pool
        std::vector<int64_t> col_off;              // slot -> [off, off + len) in pool
        std::vector<int32_t> col_len;
        std::vector<int32_t> pool;
        pool.reserve(size_t(4) * size_t(nS));
        std::vector<int32_t> cur, tmp;
        cur.reserve(64); tmp.reserve(64);
        for (int64_t r = 0; r < nS; ++r) {
            const int32_t s = sorder[r];
            if (!s_neg[s]) continue;               // clearing: a dim-2 birth kills nothing
            cur.clear();
            for (int t = 0; t < 4; ++t) {
                const int32_t e = sedge[4 * s + t];
                if (E.type[e] == 1) cur.push_back(erank[e]);   // compression
            }
            std::sort(cur.begin(), cur.end());
            while (!cur.empty()) {
                const int32_t piv = cur.back();
                const int32_t o = owner[piv];
                if (o < 0) break;
                const int32_t* c = pool.data() + col_off[o];
                tmp.clear();
                std::set_symmetric_difference(cur.begin(), cur.end(), c, c + col_len[o],
                                              std::back_inserter(tmp));
                cur.swap(tmp);
                E.n_additions++;
            }
            if (cur.empty()) { E.inconsistent++; continue; }
            const int32_t piv = cur.back();
            owner[piv] = int32_t(col_off.size());
            col_off.push_back(int64_t(pool.size()));
            col_len.push_back(int32_t(cur.size()));
            pool.insert(pool.end(), cur.begin(), cur.end());
            const int32_t e = eorder[piv];
            E.pers[e] = sval[s] - eval[e];
        }
        // every dim-1 birth edge must have died (a box has no 1-cycles)
        int64_t n_pos = 0;
        for (int64_t e = 0; e < nE; ++e) n_pos += E.type[e];
        if (int64_t(col_off.size()) != n_pos) E.inconsistent += std::llabs(int64_t(col_off.size()) - n_pos);
    }
    return E;
}

// The graph reconstruction of ComputeGraphReconstruction.cpp on the classified edges: the
// 2D kernel's, on the 3D lattice. Writes the Morse-graph voxel mask (lattice order) + stats.
static void morse_graph_from_edges_3d(const std::vector<int32_t>& vals, const MorseEdges3D& E,
                                      int32_t thr, std::vector<uint8_t>& mask, int64_t* stats) {
    const int64_t N = E.nx * E.ny * E.nz, nE = int64_t(E.a.size());
    std::vector<int32_t> deg(N + 1, 0);
    int64_t n_vf = 0, n_crit = 0, n_crit_neg = 0, n_neg = 0;
    for (int64_t e = 0; e < nE; ++e) {
        if (E.type[e] == 0) n_neg++;
        if (E.pers[e] > thr) { n_crit++; if (E.type[e] == 0) n_crit_neg++; continue; }
        if (E.type[e] == 0) { deg[E.a[e]]++; deg[E.other(e)]++; n_vf++; }
    }
    std::vector<int64_t> off(N + 1, 0);
    for (int64_t p = 0; p < N; ++p) off[p + 1] = off[p] + deg[p];
    std::vector<int32_t> adj(off[N]);
    {
        std::vector<int64_t> fill(off.begin(), off.end() - 1);
        for (int64_t e = 0; e < nE; ++e) {
            if (E.type[e] != 0 || E.pers[e] > thr) continue;
            const int32_t p = E.a[e], q = E.other(e);
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
    // (value, z, y, x) == (value, lattice index) and leaves pre[] as the BFS tree from `start`.
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
        const int32_t p = E.a[e], q = E.other(e);
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
    stats[1] = nE - n_neg;            // dim-1 birth edges
    stats[2] = n_crit;                // critical edges (pers > thr)
    stats[3] = n_crit_neg;            // ... of which dim-0 saddles
    stats[4] = n_vf;                  // vector-field edges
    stats[5] = n_components;          // vector-field trees traversed
    stats[6] = n_mask;                // Morse-graph voxels
    stats[7] = E.inconsistent;        // dual / reduction disagreements (0)
    stats[8] = E.n_dim2;              // dim-2 pairs
    stats[9] = E.n_additions;         // column additions in the dim-1 reduction
}

// C-order array (X, Y, Z) -> lattice order x + X*(y + Y*z): the released writer's im_cube[i, j, k]
// with i = array axis 0, j = axis 1, k = axis 2 (the stack index).
template <typename T>
static void to_lattice(const T* img, int64_t X, int64_t Y, int64_t Z, std::vector<T>& out) {
    out.resize(X * Y * Z);
    for (int64_t i = 0; i < X; ++i)
        for (int64_t j = 0; j < Y; ++j)
            for (int64_t k = 0; k < Z; ++k) out[i + X * (j + Y * k)] = img[(i * Y + j) * Z + k];
}

static const int N_STATS = 10;

static void morse_graph_one(const int32_t* img, int64_t X, int64_t Y, int64_t Z, int32_t thr,
                            uint8_t* mask_out, int64_t* stats) {
    std::vector<int32_t> vals;
    to_lattice(img, X, Y, Z, vals);
    MorseEdges3D E = morse_edges_3d(vals, X, Y, Z);
    std::vector<uint8_t> mask;
    morse_graph_from_edges_3d(vals, E, thr, mask, stats);
    for (int64_t i = 0; i < X; ++i)
        for (int64_t j = 0; j < Y; ++j)
            for (int64_t k = 0; k < Z; ++k) mask_out[(i * Y + j) * Z + k] = mask[i + X * (j + Y * k)];
}

template <typename F>
static void run_pool(int64_t B, int n_threads, F&& work) {
    const int T = int(std::max<int64_t>(1, std::min<int64_t>(B, std::max(1, n_threads))));
    if (T == 1) { for (int64_t b = 0; b < B; ++b) work(b); return; }
    std::atomic<int64_t> next(0);
    std::vector<std::exception_ptr> errs(T);
    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t]() {
            try {
                for (int64_t b = next++; b < B; b = next++) work(b);
            } catch (...) { errs[t] = std::current_exception(); }
        });
    for (auto& th : pool) th.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
}

static py::tuple py_morse_graph_batch(py::array_t<int32_t, py::array::c_style | py::array::forcecast> vals,
                                      int32_t threshold, int n_threads) {
    if (vals.ndim() != 4) throw std::invalid_argument("vals must be (B, X, Y, Z) int32");
    const int64_t B = vals.shape(0), X = vals.shape(1), Y = vals.shape(2), Z = vals.shape(3);
    py::array_t<uint8_t> mask({B, X, Y, Z});
    py::array_t<int64_t> stats({B, int64_t(N_STATS)});
    const int32_t* vp = vals.data();
    uint8_t* mp = mask.mutable_data();
    int64_t* sp = stats.mutable_data();
    {
        py::gil_scoped_release release;
        run_pool(B, n_threads, [&](int64_t b) {
            morse_graph_one(vp + b * X * Y * Z, X, Y, Z, threshold, mp + b * X * Y * Z, sp + b * N_STATS);
        });
    }
    return py::make_tuple(mask, stats);
}

// Edge-level output for the parity test: (a, b, pers, type, inconsistent, n_dim2) with a, b
// lattice indices (x + X*(y + Y*z)), in DIPHA cell-index order.
static py::tuple py_morse_edges(py::array_t<int32_t, py::array::c_style | py::array::forcecast> vals) {
    if (vals.ndim() != 3) throw std::invalid_argument("vals must be (X, Y, Z) int32");
    const int64_t X = vals.shape(0), Y = vals.shape(1), Z = vals.shape(2);
    std::vector<int32_t> lat;
    to_lattice(vals.data(), X, Y, Z, lat);
    MorseEdges3D E = morse_edges_3d(lat, X, Y, Z);
    const int64_t nE = int64_t(E.a.size());
    py::array_t<int64_t> a(nE), b(nE), pers(nE);
    py::array_t<uint8_t> type(nE);
    auto A = a.mutable_unchecked<1>();
    auto Bq = b.mutable_unchecked<1>();
    auto P = pers.mutable_unchecked<1>();
    auto T = type.mutable_unchecked<1>();
    for (int64_t e = 0; e < nE; ++e) {
        A(e) = E.a[e]; Bq(e) = E.other(e); P(e) = E.pers[e]; T(e) = E.type[e];
    }
    return py::make_tuple(a, b, pers, type, E.inconsistent, E.n_dim2);
}

}  // namespace

PYBIND11_MODULE(dmt3d_core, mod) {
    mod.doc() = "DMT (ICLR 2021) Morse-graph kernel on volumes, pinned to the released morse_DMT pipeline";
    mod.def("morse_graph_batch", &py_morse_graph_batch, py::arg("vals"), py::arg("threshold"),
            py::arg("n_threads") = 8,
            "Morse-graph masks (B,X,Y,Z) uint8 + stats (B,10) int64 of quantised likelihoods (B,X,Y,Z) int32");
    mod.def("morse_edges", &py_morse_edges, py::arg("vals"),
            "Per-edge (a, b, pers, type, inconsistent, n_dim2) of one quantised volume (X,Y,Z) int32");
    mod.attr("N_STATS") = N_STATS;
}
