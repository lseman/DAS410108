// decision_tree.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// =============================
// Utilities
// =============================
struct Node {
    int feature = -1;
    double threshold = 0.0;
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;

    // leaf payload
    double value = 0.0;                 // regression mean or class index (0-based)
    std::vector<int> class_counts;      // for classification leaves

    // diagnostics
    int n_samples = 0;
    double impurity = 0.0;

    bool is_leaf() const { return !left && !right; }
};

inline double gini_impurity_from_counts(const std::vector<int>& counts) {
    long long total = 0;
    for (int c : counts) total += c;
    if (total == 0) return 0.0;
    double inv = 1.0 / static_cast<double>(total);
    double sumsq = 0.0;
    for (int c : counts) {
        double p = c * inv;
        sumsq += p * p;
    }
    return 1.0 - sumsq;
}

inline double entropy_from_counts(const std::vector<int>& counts) {
    long long total = 0;
    for (int c : counts) total += c;
    if (total == 0) return 0.0;
    double inv = 1.0 / static_cast<double>(total);
    double ent = 0.0;
    for (int c : counts) {
        if (c > 0) {
            double p = c * inv;
            ent -= p * std::log2(p);
        }
    }
    return ent;
}

inline double sse_from_sums(double n, double s1, double s2) {
    if (n <= 0.0) return 0.0;
    return s2 - (s1 * s1) / n;
}

struct SplitResult {
    int feature = -1;
    double threshold = 0.0;
    double cost = std::numeric_limits<double>::infinity(); // weighted child impurity
    std::vector<int> left_idx;
    std::vector<int> right_idx;
    bool valid = false;
};

// =============================
// FastDecisionTree
// =============================
class FastDecisionTree {
public:
    FastDecisionTree(const std::string& criterion = "gini",
                     int max_depth = -1,
                     int min_samples_split = 2,
                     int min_samples_leaf = 1,
                     double min_impurity_decrease = 0.0,
                     bool normalize_gain = true,
                     bool use_histogram = false,
                     int n_bins = 32,
                     const std::string& binning = "quantile")
        : criterion_(criterion),
          max_depth_(max_depth > 0 ? max_depth : 1000000),
          min_samples_split_(min_samples_split),
          min_samples_leaf_(min_samples_leaf),
          min_impurity_decrease_(min_impurity_decrease),
          normalize_gain_(normalize_gain),
          use_histogram_(use_histogram),
          n_bins_(n_bins),
          binning_(binning) {
        if (!(criterion_ == "gini" || criterion_ == "entropy" || criterion_ == "mse"))
            throw std::invalid_argument("criterion must be 'gini', 'entropy', or 'mse'");
        is_classification_ = (criterion_ != "mse");
        if (!(binning_ == "quantile" || binning_ == "uniform"))
            throw std::invalid_argument("binning must be 'quantile' or 'uniform'");
        if (n_bins_ < 2) n_bins_ = 2;
    }

    void fit(py::array_t<double> X, py::array y) {
        auto Xb = X.request();
        if (Xb.ndim != 2) throw std::runtime_error("X must be 2D [n_samples, n_features]");
        n_samples_ = static_cast<int>(Xb.shape[0]);
        n_features_ = static_cast<int>(Xb.shape[1]);

        // ingest y and set up classification/regression maps
        if (is_classification_) {
            read_y_classification_(y);
        } else {
            read_y_regression_(y);
        }

        // Precompute global bin edges if requested
        bins_.clear();
        if (use_histogram_) {
            prepare_bins_(X);
        }

        std::vector<int> idx(n_samples_);
        std::iota(idx.begin(), idx.end(), 0);

        if (is_classification_) {
            py::array_t<int> y_int(n_samples_);
            auto yb = y_int.request();
            int* yp = static_cast<int*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) yp[i] = y_int_labels_[i];
            root_ = grow_classification_(X, y_int, idx, 0);
        } else {
            py::array_t<double> y_d(n_samples_);
            auto yb = y_d.request();
            double* yp = static_cast<double*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) yp[i] = y_double_[i];
            root_ = grow_regression_(X, y_d, idx, 0);
        }
    }

    std::vector<double> predict(py::array_t<double> X) {
        if (!root_) throw std::runtime_error("Call fit() before predict().");
        auto Xb = X.request();
        if (Xb.ndim != 2 || Xb.shape[1] != n_features_)
            throw std::runtime_error("X has wrong shape for predict().");
        double* Xp = static_cast<double*>(Xb.ptr);
        int ns = static_cast<int>(Xb.shape[0]);

        std::vector<double> out;
        out.reserve(ns);
        for (int i = 0; i < ns; ++i) {
            double* xi = Xp + i * n_features_;
            out.push_back(predict_single_(xi, root_.get()));
        }
        return out;
    }

    std::vector<std::vector<double>> predict_proba(py::array_t<double> X) {
        if (!is_classification_)
            throw std::runtime_error("predict_proba only for classification.");
        if (!root_) throw std::runtime_error("Call fit() before predict_proba().");

        auto Xb = X.request();
        if (Xb.ndim != 2 || Xb.shape[1] != n_features_)
            throw std::runtime_error("X has wrong shape for predict_proba().");
        double* Xp = static_cast<double*>(Xb.ptr);
        int ns = static_cast<int>(Xb.shape[0]);

        std::vector<std::vector<double>> probs;
        probs.reserve(ns);
        for (int i = 0; i < ns; ++i) {
            double* xi = Xp + i * n_features_;
            Node* leaf = descend_(xi, root_.get());
            std::vector<double> p(n_classes_, 0.0);
            int tot = 0;
            for (int c : leaf->class_counts) tot += c;
            if (tot > 0) {
                for (int k = 0; k < n_classes_; ++k)
                    p[k] = static_cast<double>(leaf->class_counts[k]) / tot;
            } else {
                double u = 1.0 / n_classes_;
                std::fill(p.begin(), p.end(), u);
            }
            probs.push_back(std::move(p));
        }
        return probs;
    }

private:
    // --------------------------
    // Data ingestion
    // --------------------------
    void read_y_classification_(py::array& y) {
        auto yb = y.request();
        if (yb.ndim != 1) throw std::runtime_error("y must be 1D");
        if (static_cast<int>(yb.shape[0]) != n_samples_)
            throw std::runtime_error("X,y size mismatch");

        // collect labels, then map to 0..K-1
        std::set<double> uniq;
        y_int_labels_.resize(n_samples_);
        if (py::dtype::of<int>().is(y.dtype())) {
            int* yp = static_cast<int*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) uniq.insert(static_cast<double>(yp[i]));
            classes_.assign(uniq.begin(), uniq.end());
            n_classes_ = static_cast<int>(classes_.size());
            std::map<double, int> to_idx;
            for (int i = 0; i < n_classes_; ++i) to_idx[classes_[i]] = i;
            for (int i = 0; i < n_samples_; ++i) y_int_labels_[i] = to_idx[static_cast<double>(yp[i])];
        } else {
            // assume float/double
            double* yp = static_cast<double*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) uniq.insert(yp[i]);
            classes_.assign(uniq.begin(), uniq.end());
            n_classes_ = static_cast<int>(classes_.size());
            std::map<double, int> to_idx;
            for (int i = 0; i < n_classes_; ++i) to_idx[classes_[i]] = i;
            for (int i = 0; i < n_samples_; ++i) y_int_labels_[i] = to_idx[yp[i]];
        }
    }

    void read_y_regression_(py::array& y) {
        auto yb = y.request();
        if (yb.ndim != 1) throw std::runtime_error("y must be 1D");
        if (static_cast<int>(yb.shape[0]) != n_samples_)
            throw std::runtime_error("X,y size mismatch");
        y_double_.resize(n_samples_);
        if (py::dtype::of<double>().is(y.dtype())) {
            double* yp = static_cast<double*>(yb.ptr);
            std::copy(yp, yp + n_samples_, y_double_.begin());
        } else if (py::dtype::of<float>().is(y.dtype())) {
            float* yp = static_cast<float*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) y_double_[i] = static_cast<double>(yp[i]);
        } else if (py::dtype::of<int>().is(y.dtype())) {
            int* yp = static_cast<int*>(yb.ptr);
            for (int i = 0; i < n_samples_; ++i) y_double_[i] = static_cast<double>(yp[i]);
        } else {
            throw std::runtime_error("Unsupported y dtype for regression");
        }
    }

    // --------------------------
    // Binning (global, per feature)
    // --------------------------
    void prepare_bins_(py::array_t<double>& X) {
        auto Xb = X.request();
        double* Xp = static_cast<double*>(Xb.ptr);
        bins_.resize(n_features_);

        for (int j = 0; j < n_features_; ++j) {
            // collect non-NaN values
            std::vector<double> col;
            col.reserve(n_samples_);
            for (int i = 0; i < n_samples_; ++i) {
                double v = Xp[i * n_features_ + j];
                if (!std::isnan(v)) col.push_back(v);
            }
            if (col.empty()) {
                // degenerate feature
                bins_[j] = { -INFINITY, INFINITY };
                continue;
            }
            std::sort(col.begin(), col.end());

            std::vector<double> edges;
            edges.reserve(n_bins_ + 1);
            if (binning_ == "uniform") {
                double mn = col.front(), mx = col.back();
                if (mn == mx) {
                    edges = { mn, mn + 1e-12 };
                } else {
                    for (int b = 0; b <= n_bins_; ++b) {
                        double t = mn + (mx - mn) * (static_cast<double>(b) / n_bins_);
                        edges.push_back(t);
                    }
                }
            } else { // quantile
                for (int b = 0; b <= n_bins_; ++b) {
                    double q = static_cast<double>(b) / n_bins_;
                    double pos = q * (col.size() - 1);
                    size_t lo = static_cast<size_t>(std::floor(pos));
                    size_t hi = static_cast<size_t>(std::ceil(pos));
                    double v = (lo == hi) ? col[lo] : (col[lo] * (hi - pos) + col[hi] * (pos - lo));
                    if (edges.empty() || v != edges.back()) edges.push_back(v);
                }
                if (edges.size() < 2) {
                    double v = edges.empty() ? col[0] : edges[0];
                    edges = { v - 1e-12, v + 1e-12 };
                }
            }
            // ensure monotone and at least two edges
            if (edges.size() < 2) edges = { edges[0] - 1e-12, edges[0] + 1e-12 };
            bins_[j] = std::move(edges);
        }
    }

    inline int digitize_(double x, const std::vector<double>& edges) const {
        // bins are [edges[k], edges[k+1]), k = 0..B-1; NaN -> last bin (right)
        int B = static_cast<int>(edges.size()) - 1;
        if (std::isnan(x)) return B - 1; // route NaN to right-most bin
        // binary search upper_bound on edges[1..B-1]
        int lo = 1, hi = B - 1, ans = B - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (x < edges[mid]) { ans = mid - 1; hi = mid - 1; }
            else { lo = mid + 1; }
        }
        return ans;
    }

    // --------------------------
    // Tree growth
    // --------------------------
    std::unique_ptr<Node> grow_regression_(py::array_t<double>& X,
                                           py::array_t<double>& y,
                                           const std::vector<int>& idx,
                                           int depth) {
        auto node = std::make_unique<Node>();
        node->n_samples = static_cast<int>(idx.size());

        auto yb = y.request();
        double* yp = static_cast<double*>(yb.ptr);

        // stats
        double sum = 0.0, sumsq = 0.0;
        for (int id : idx) { sum += yp[id]; sumsq += yp[id] * yp[id]; }
        node->value = (idx.empty() ? 0.0 : sum / idx.size());
        node->impurity = sse_from_sums(static_cast<double>(idx.size()), sum, sumsq);

        // stop?
        if (depth >= max_depth_ || static_cast<int>(idx.size()) < min_samples_split_ ||
            static_cast<int>(idx.size()) < 2 * min_samples_leaf_) {
            return node;
        }

        SplitResult split = use_histogram_
                            ? best_split_hist_reg_(X, y, idx)
                            : best_split_exact_reg_(X, y, idx);

        if (!split.valid) return node;

        double decrease = node->impurity - split.cost;
        if (normalize_gain_) decrease /= std::max(1, node->n_samples);
        if (decrease < min_impurity_decrease_) return node;

        node->feature = split.feature;
        node->threshold = split.threshold;
        node->left = grow_regression_(X, y, split.left_idx, depth + 1);
        node->right = grow_regression_(X, y, split.right_idx, depth + 1);
        return node;
    }

    std::unique_ptr<Node> grow_classification_(py::array_t<double>& X,
                                               py::array_t<int>& y,
                                               const std::vector<int>& idx,
                                               int depth) {
        auto node = std::make_unique<Node>();
        node->n_samples = static_cast<int>(idx.size());
        node->class_counts.assign(n_classes_, 0);

        auto yb = y.request();
        int* yp = static_cast<int*>(yb.ptr);
        for (int id : idx) node->class_counts[yp[id]]++;

        node->impurity = (criterion_ == "gini")
                         ? gini_impurity_from_counts(node->class_counts)
                         : entropy_from_counts(node->class_counts);

        // majority class (store as 0-based index; map to label at predict)
        node->value = static_cast<double>(
            std::distance(node->class_counts.begin(),
                          std::max_element(node->class_counts.begin(), node->class_counts.end()))
        );

        if (depth >= max_depth_ || static_cast<int>(idx.size()) < min_samples_split_ ||
            static_cast<int>(idx.size()) < 2 * min_samples_leaf_) {
            return node;
        }

        SplitResult split = use_histogram_
                            ? best_split_hist_cls_(X, y, idx)
                            : best_split_exact_cls_(X, y, idx);

        if (!split.valid) return node;

        double decrease = node->impurity - split.cost / std::max(1, node->n_samples); // cost already weighted
        // For classification we computed cost = nL*impL + nR*impR; parent impurity is unweighted.
        // To compare apples-to-apples: convert parent to weighted (multiply by n)
        double parent_weighted = node->impurity * node->n_samples;
        decrease = parent_weighted - split.cost;
        if (normalize_gain_) decrease /= std::max(1, node->n_samples);
        if (decrease < min_impurity_decrease_) return node;

        node->feature = split.feature;
        node->threshold = split.threshold;
        node->left = grow_classification_(X, y, split.left_idx, depth + 1);
        node->right = grow_classification_(X, y, split.right_idx, depth + 1);
        return node;
    }

    // --------------------------
    // Exact splits
    // --------------------------
    SplitResult best_split_exact_reg_(py::array_t<double>& X,
                                      py::array_t<double>& y,
                                      const std::vector<int>& idx) {
        auto Xb = X.request();
        double* Xp = static_cast<double*>(Xb.ptr);
        int ns = static_cast<int>(idx.size());

        auto yb = y.request();
        double* yp = static_cast<double*>(yb.ptr);

        // totals
        double total_s1 = 0.0, total_s2 = 0.0;
        for (int id : idx) { total_s1 += yp[id]; total_s2 += yp[id] * yp[id]; }

        SplitResult best;
        for (int j = 0; j < n_features_; ++j) {
            std::vector<std::pair<double,int>> vals;
            vals.reserve(ns);
            for (int id : idx) vals.emplace_back(Xp[id * n_features_ + j], id);
            std::stable_sort(vals.begin(), vals.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });

            double left_s1 = 0.0, left_s2 = 0.0;
            int left_n = 0;
            for (int i = 0; i < ns - 1; ++i) {
                int id = vals[i].second;
                double yv = yp[id];
                left_s1 += yv; left_s2 += yv * yv; left_n++;
                int right_n = ns - left_n;
                if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;
                if (vals[i].first >= vals[i+1].first) continue; // not a valid split

                double right_s1 = total_s1 - left_s1;
                double right_s2 = total_s2 - left_s2;

                double cost = sse_from_sums(left_n, left_s1, left_s2)
                            + sse_from_sums(right_n, right_s1, right_s2);
                if (cost < best.cost) {
                    best.cost = cost;
                    best.feature = j;
                    best.threshold = 0.5 * (vals[i].first + vals[i+1].first);
                    best.valid = true;

                    best.left_idx.clear();
                    best.right_idx.clear();
                    best.left_idx.reserve(left_n);
                    best.right_idx.reserve(right_n);
                    for (int t = 0; t <= i; ++t) best.left_idx.push_back(vals[t].second);
                    for (int t = i+1; t < ns; ++t) best.right_idx.push_back(vals[t].second);
                }
            }
        }
        return best;
    }

    SplitResult best_split_exact_cls_(py::array_t<double>& X,
                                      py::array_t<int>& y,
                                      const std::vector<int>& idx) {
        auto Xb = X.request();
        double* Xp = static_cast<double*>(Xb.ptr);
        int ns = static_cast<int>(idx.size());

        auto yb = y.request();
        int* yp = static_cast<int*>(yb.ptr);

        // total counts
        std::vector<int> total(n_classes_, 0);
        for (int id : idx) total[yp[id]]++;

        SplitResult best;
        for (int j = 0; j < n_features_; ++j) {
            std::vector<std::pair<double,int>> vals;
            vals.reserve(ns);
            for (int id : idx) vals.emplace_back(Xp[id * n_features_ + j], id);
            std::stable_sort(vals.begin(), vals.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });

            std::vector<int> left_counts(n_classes_, 0);
            int left_n = 0;

            for (int i = 0; i < ns - 1; ++i) {
                int id = vals[i].second;
                left_counts[yp[id]]++;
                left_n++;
                int right_n = ns - left_n;
                if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;
                if (vals[i].first >= vals[i+1].first) continue;

                std::vector<int> right_counts(n_classes_);
                for (int c = 0; c < n_classes_; ++c) right_counts[c] = total[c] - left_counts[c];

                double impL = (criterion_ == "gini")
                              ? gini_impurity_from_counts(left_counts)
                              : entropy_from_counts(left_counts);
                double impR = (criterion_ == "gini")
                              ? gini_impurity_from_counts(right_counts)
                              : entropy_from_counts(right_counts);

                double cost = left_n * impL + right_n * impR; // weighted child impurity
                if (cost < best.cost) {
                    best.cost = cost;
                    best.feature = j;
                    best.threshold = 0.5 * (vals[i].first + vals[i+1].first);
                    best.valid = true;

                    best.left_idx.clear();
                    best.right_idx.clear();
                    best.left_idx.reserve(left_n);
                    best.right_idx.reserve(right_n);
                    for (int t = 0; t <= i; ++t) best.left_idx.push_back(vals[t].second);
                    for (int t = i+1; t < ns; ++t) best.right_idx.push_back(vals[t].second);
                }
            }
        }
        return best;
    }

    // --------------------------
    // Histogram splits
    // --------------------------
    SplitResult best_split_hist_reg_(py::array_t<double>& X,
                                     py::array_t<double>& y,
                                     const std::vector<int>& idx) {
        auto Xb = X.request();
        double* Xp = static_cast<double*>(Xb.ptr);
        auto yb = y.request();
        double* yp = static_cast<double*>(yb.ptr);
        int ns = static_cast<int>(idx.size());

        // totals
        double total_s1 = 0.0, total_s2 = 0.0;
        for (int id : idx) { total_s1 += yp[id]; total_s2 += yp[id] * yp[id]; }

        SplitResult best;

        for (int j = 0; j < n_features_; ++j) {
            const auto& edges = bins_[j];
            int B = static_cast<int>(edges.size()) - 1;
            if (B <= 1) continue;

            // per-bin stats
            std::vector<double> s1(B, 0.0), s2(B, 0.0);
            std::vector<double> cnt(B, 0.0);

            for (int id : idx) {
                double v = Xp[id * n_features_ + j];
                int b = digitize_(v, edges);
                if (b < 0) b = 0;
                if (b >= B) b = B - 1;
                double yv = yp[id];
                s1[b] += yv;
                s2[b] += yv * yv;
                cnt[b] += 1.0;
            }

            // cumulative
            std::vector<double> cs1(B, 0.0), cs2(B, 0.0), csz(B, 0.0);
            cs1[0] = s1[0]; cs2[0] = s2[0]; csz[0] = cnt[0];
            for (int b = 1; b < B; ++b) {
                cs1[b] = cs1[b-1] + s1[b];
                cs2[b] = cs2[b-1] + s2[b];
                csz[b] = csz[b-1] + cnt[b];
            }

            for (int t = 0; t < B - 1; ++t) {
                double left_n = csz[t];
                double right_n = ns - left_n;
                if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;

                double ls1 = cs1[t], ls2 = cs2[t];
                double rs1 = total_s1 - ls1, rs2 = total_s2 - ls2;

                double cost = sse_from_sums(left_n, ls1, ls2)
                            + sse_from_sums(right_n, rs1, rs2);

                if (cost < best.cost) {
                    best.cost = cost;
                    // threshold between bin t and t+1: midpoint of edges
                    double thr = 0.5 * (edges[t+1] + edges[t+2]); // safe since t<=B-2
                    best.threshold = thr;
                    best.feature = j;
                    best.valid = true;

                    // materialize indices by threshold (NaN -> right)
                    best.left_idx.clear();
                    best.right_idx.clear();
                    for (int id2 : idx) {
                        double v2 = Xp[id2 * n_features_ + j];
                        bool go_left = (!std::isnan(v2) && v2 <= thr);
                        if (go_left) best.left_idx.push_back(id2);
                        else best.right_idx.push_back(id2);
                    }
                }
            }
        }
        return best;
    }

    SplitResult best_split_hist_cls_(py::array_t<double>& X,
                                     py::array_t<int>& y,
                                     const std::vector<int>& idx) {
        auto Xb = X.request();
        double* Xp = static_cast<double*>(Xb.ptr);
        auto yb = y.request();
        int* yp = static_cast<int*>(yb.ptr);
        int ns = static_cast<int>(idx.size());

        SplitResult best;

        for (int j = 0; j < n_features_; ++j) {
            const auto& edges = bins_[j];
            int B = static_cast<int>(edges.size()) - 1;
            if (B <= 1) continue;

            // counts per bin per class [B,K]
            std::vector<int> counts(B * n_classes_, 0);
            auto at = [&](int b, int c)->int& { return counts[b * n_classes_ + c]; };

            for (int id : idx) {
                double v = Xp[id * n_features_ + j];
                int b = digitize_(v, edges);
                if (b < 0) b = 0;
                if (b >= B) b = B - 1;
                at(b, yp[id]) += 1;
            }

            // cumulative over bins
            std::vector<int> cum(B * n_classes_, 0);
            for (int c = 0; c < n_classes_; ++c) cum[c] = counts[c];
            for (int b = 1; b < B; ++b) {
                for (int c = 0; c < n_classes_; ++c)
                    cum[b * n_classes_ + c] = cum[(b-1) * n_classes_ + c] + counts[b * n_classes_ + c];
            }
            std::vector<int> total(n_classes_, 0);
            for (int c = 0; c < n_classes_; ++c) total[c] = cum[(B-1) * n_classes_ + c];

            std::vector<int> binsz(B, 0), cumsz(B, 0);
            for (int b = 0; b < B; ++b) {
                int s = 0;
                for (int c = 0; c < n_classes_; ++c) s += counts[b * n_classes_ + c];
                binsz[b] = s;
                cumsz[b] = (b == 0 ? s : cumsz[b-1] + s);
            }

            for (int t = 0; t < B - 1; ++t) {
                int left_n = cumsz[t];
                int right_n = ns - left_n;
                if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;

                std::vector<int> left_counts(n_classes_, 0), right_counts(n_classes_, 0);
                for (int c = 0; c < n_classes_; ++c) {
                    left_counts[c] = cum[t * n_classes_ + c];
                    right_counts[c] = total[c] - left_counts[c];
                }

                double impL = (criterion_ == "gini")
                              ? gini_impurity_from_counts(left_counts)
                              : entropy_from_counts(left_counts);
                double impR = (criterion_ == "gini")
                              ? gini_impurity_from_counts(right_counts)
                              : entropy_from_counts(right_counts);

                double cost = left_n * impL + right_n * impR;
                if (cost < best.cost) {
                    best.cost = cost;
                    double thr = 0.5 * (edges[t+1] + edges[t+2]);
                    best.threshold = thr;
                    best.feature = j;
                    best.valid = true;

                    best.left_idx.clear();
                    best.right_idx.clear();
                    for (int id2 : idx) {
                        double v2 = Xp[id2 * n_features_ + j];
                        bool go_left = (!std::isnan(v2) && v2 <= thr);
                        if (go_left) best.left_idx.push_back(id2);
                        else best.right_idx.push_back(id2);
                    }
                }
            }
        }
        return best;
    }

    // --------------------------
    // Prediction helpers
    // --------------------------
    inline Node* descend_(const double* x, Node* n) const {
        while (!n->is_leaf()) {
            double v = x[n->feature];
            bool left = (!std::isnan(v) && v <= n->threshold);
            n = left ? n->left.get() : n->right.get();
        }
        return n;
    }

    inline double predict_single_(const double* x, Node* n) const {
        n = descend_(x, n);
        if (is_classification_) {
            int cls_idx = static_cast<int>(n->value);
            // map back to original label
            return classes_[cls_idx];
        }
        return n->value;
    }

private:
    // config
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    bool normalize_gain_;
    bool use_histogram_;
    int n_bins_;
    std::string binning_;

    // data dims and flags
    int n_features_ = 0;
    int n_samples_ = 0;
    bool is_classification_ = true;

    // label maps
    std::vector<double> classes_;     // original labels (size K)
    int n_classes_ = 0;
    std::vector<int> y_int_labels_;   // 0..K-1
    std::vector<double> y_double_;    // regression target

    // global bins per feature (edges)
    std::vector<std::vector<double>> bins_;

    // root
    std::unique_ptr<Node> root_ = nullptr;
};

// =============================
// pybind11 bindings
// =============================
PYBIND11_MODULE(fast_tree, m) {
    m.doc() = "Fast C++ Decision Tree with exact and histogram splitters";

    py::class_<FastDecisionTree>(m, "FastDecisionTree")
        .def(py::init<const std::string&, int, int, int, double, bool, bool, int, const std::string&>(),
             py::arg("criterion") = "gini",
             py::arg("max_depth") = -1,
             py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1,
             py::arg("min_impurity_decrease") = 0.0,
             py::arg("normalize_gain") = true,
             py::arg("use_histogram") = false,
             py::arg("n_bins") = 32,
             py::arg("binning") = "quantile")
        .def("fit", &FastDecisionTree::fit, "Fit the tree")
        .def("predict", &FastDecisionTree::predict, "Predict labels or values")
        .def("predict_proba", &FastDecisionTree::predict_proba, "Predict class probabilities");

    // Small utility exports (optional)
    m.def("gini_impurity_from_counts", &gini_impurity_from_counts);
    m.def("entropy_from_counts", &entropy_from_counts);
}
