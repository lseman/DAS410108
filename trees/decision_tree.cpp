// decision_tree.cpp (fast_tree)

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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

#ifndef NO_OMP
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace py = pybind11;

// =============================
// Utilities
// =============================
struct Node {
  int feature = -1;
  double threshold = 0.0;
  std::unique_ptr<Node> left = nullptr;
  std::unique_ptr<Node> right = nullptr;

  double value = 0.0;            // regression mean or class index (0-based)
  std::vector<int> class_counts; // for classification leaves

  int n_samples = 0;
  double impurity = 0.0;

  bool is_leaf() const { return !left && !right; }
};

inline double gini_impurity_from_counts(const std::vector<int> &counts) {
  long long total = 0;
  for (int c : counts) total += c;
  if (total == 0) return 0.0;
  const double inv = 1.0 / static_cast<double>(total);
  double sumsq = 0.0;
  for (int c : counts) {
    const double p = c * inv;
    sumsq += p * p;
  }
  return 1.0 - sumsq;
}

inline double entropy_from_counts(const std::vector<int> &counts) {
  long long total = 0;
  for (int c : counts) total += c;
  if (total == 0) return 0.0;
  const double inv = 1.0 / static_cast<double>(total);
  double ent = 0.0;
  for (int c : counts) if (c > 0) {
    const double p = c * inv;
    ent -= p * std::log2(p);
  }
  return ent;
}

inline double sse_from_sums(double n, double s1, double s2) {
  if (n <= 0.0) return 0.0;
  double sse = s2 - (s1 * s1) / n;
  return sse < 0.0 ? 0.0 : sse;
}

struct SplitResult {
  int feature = -1;
  double threshold = 0.0;
  double cost = std::numeric_limits<double>::infinity(); // weighted child impurity for cls; sum SSE for reg
  int left_count = 0;  // count only
  bool valid = false;
};

inline bool is_better(const SplitResult &a, const SplitResult &b) {
  if (!a.valid) return false;
  if (!b.valid) return true;
  if (a.cost < b.cost) return true;
  if (a.cost > b.cost) return false;
  if (a.feature < b.feature) return true;
  if (a.feature > b.feature) return false;
  return a.threshold < b.threshold;
}

// =============================
// FastDecisionTree
// =============================
class FastDecisionTree {
public:
  FastDecisionTree(const std::string &criterion = "gini",
                   int max_depth = -1,
                   int min_samples_split = 2,
                   int min_samples_leaf = 1,
                   double min_impurity_decrease = 0.0,
                   bool normalize_gain = true,
                   bool use_histogram = false,
                   int n_bins = 32,
                   const std::string &binning = "quantile")
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
    n_samples_  = static_cast<int>(Xb.shape[0]);
    n_features_ = static_cast<int>(Xb.shape[1]);
    X_ptr_ = static_cast<const double *>(Xb.ptr); // raw pointer to X

    if (is_classification_) read_y_classification_(y);
    else                    read_y_regression_(y);

    bins_.clear();
    if (use_histogram_) prepare_bins_(X_ptr_);

    std::vector<int> idx(n_samples_);
    std::iota(idx.begin(), idx.end(), 0);

    root_ = is_classification_ ? grow_classification_(idx, 0)
                               : grow_regression_(idx, 0);
  }

  std::vector<double> predict(py::array_t<double> X) {
    if (!root_) throw std::runtime_error("Call fit() before predict().");
    auto Xb = X.request();
    if (Xb.ndim != 2 || static_cast<int>(Xb.shape[1]) != n_features_)
      throw std::runtime_error("X has wrong shape for predict().");
    const double *Xp = static_cast<const double *>(Xb.ptr);
    const int ns = static_cast<int>(Xb.shape[0]);

    std::vector<double> out;
    out.reserve(ns);
    for (int i = 0; i < ns; ++i)
      out.push_back(predict_single_(Xp + i * n_features_, root_.get()));
    return out;
  }

  std::vector<std::vector<double>> predict_proba(py::array_t<double> X) {
    if (!is_classification_) throw std::runtime_error("predict_proba only for classification.");
    if (!root_) throw std::runtime_error("Call fit() before predict_proba().");

    auto Xb = X.request();
    if (Xb.ndim != 2 || static_cast<int>(Xb.shape[1]) != n_features_)
      throw std::runtime_error("X has wrong shape for predict_proba().");
    const double *Xp = static_cast<const double *>(Xb.ptr);
    const int ns = static_cast<int>(Xb.shape[0]);

    std::vector<std::vector<double>> probs;
    probs.reserve(ns);
    for (int i = 0; i < ns; ++i) {
      Node *leaf = descend_(Xp + i * n_features_, root_.get());
      std::vector<double> p(n_classes_, 0.0);
      int tot = 0; for (int c : leaf->class_counts) tot += c;
      if (tot > 0) {
        for (int k = 0; k < n_classes_; ++k)
          p[k] = static_cast<double>(leaf->class_counts[k]) / tot;
      } else {
        const double u = 1.0 / std::max(1, n_classes_);
        std::fill(p.begin(), p.end(), u);
      }
      probs.push_back(std::move(p));
    }
    return probs;
  }

private:
  // --------------------------
  // Data ingestion (compact)
  // --------------------------
  void read_y_classification_(py::array &y) {
    auto yb = y.request();
    if (yb.ndim != 1) throw std::runtime_error("y must be 1D");
    if (static_cast<int>(yb.shape[0]) != n_samples_) throw std::runtime_error("X,y size mismatch");

    std::set<double> uniq;
    classes_.clear();

    y_int_labels_.resize(n_samples_);
    if (py::dtype::of<int>().is(y.dtype())) {
      const int *yp = static_cast<const int *>(yb.ptr);
      for (int i = 0; i < n_samples_; ++i) uniq.insert(static_cast<double>(yp[i]));
      classes_.assign(uniq.begin(), uniq.end());
      n_classes_ = static_cast<int>(classes_.size());
      std::map<double,int> to_idx;
      for (int i = 0; i < n_classes_; ++i) to_idx[classes_[i]] = i;
      for (int i = 0; i < n_samples_; ++i) y_int_labels_[i] = to_idx[static_cast<double>(yp[i])];
    } else {
      const double *yp = static_cast<const double *>(yb.ptr);
      for (int i = 0; i < n_samples_; ++i) uniq.insert(yp[i]);
      classes_.assign(uniq.begin(), uniq.end());
      n_classes_ = static_cast<int>(classes_.size());
      std::map<double,int> to_idx;
      for (int i = 0; i < n_classes_; ++i) to_idx[classes_[i]] = i;
      for (int i = 0; i < n_samples_; ++i) y_int_labels_[i] = to_idx[yp[i]];
    }
  }

  void read_y_regression_(py::array &y) {
    auto yb = y.request();
    if (yb.ndim != 1) throw std::runtime_error("y must be 1D");
    if (static_cast<int>(yb.shape[0]) != n_samples_) throw std::runtime_error("X,y size mismatch");
    y_double_.resize(n_samples_);
    if (py::dtype::of<double>().is(y.dtype())) {
      const double *yp = static_cast<const double *>(yb.ptr);
      std::copy(yp, yp + n_samples_, y_double_.begin());
    } else if (py::dtype::of<float>().is(y.dtype())) {
      const float *yp = static_cast<const float *>(yb.ptr);
      for (int i = 0; i < n_samples_; ++i) y_double_[i] = static_cast<double>(yp[i]);
    } else if (py::dtype::of<int>().is(y.dtype())) {
      const int *yp = static_cast<const int *>(yb.ptr);
      for (int i = 0; i < n_samples_; ++i) y_double_[i] = static_cast<double>(yp[i]);
    } else {
      throw std::runtime_error("Unsupported y dtype for regression");
    }
  }

  // --------------------------
  // Binning (global, per feature)
  // --------------------------
  void prepare_bins_(const double *Xp) {
    bins_.resize(n_features_);
    for (int j = 0; j < n_features_; ++j) {
      std::vector<double> col;
      col.reserve(n_samples_);
      for (int i = 0; i < n_samples_; ++i) {
        double v = Xp[i * n_features_ + j];
        if (!std::isnan(v)) col.push_back(v);
      }
      if (col.empty()) { bins_[j] = { -INFINITY, INFINITY }; continue; }
      std::sort(col.begin(), col.end());

      std::vector<double> edges;
      edges.reserve(n_bins_ + 1);
      if (binning_ == "uniform") {
        double mn = col.front(), mx = col.back();
        if (mn == mx) edges = { mn, mn + 1e-12 };
        else {
          for (int b = 0; b <= n_bins_; ++b) {
            double t = mn + (mx - mn) * (double(b) / n_bins_);
            edges.push_back(t);
          }
        }
      } else { // quantile
        const int N = static_cast<int>(col.size());
        for (int b = 0; b <= n_bins_; ++b) {
          double q = double(b) / n_bins_;
          double pos = q * (N - 1);
          int lo = (int)std::floor(pos);
          int hi = (int)std::ceil(pos);
          double v = (lo == hi) ? col[lo] : (col[lo] * (hi - pos) + col[hi] * (pos - lo));
          if (edges.empty() || v != edges.back()) edges.push_back(v);
        }
        if (edges.size() < 2) {
          double v = edges.empty() ? col[0] : edges[0];
          edges = { v - 1e-12, v + 1e-12 };
        }
      }
      if (edges.size() < 2) edges = { edges[0] - 1e-12, edges[0] + 1e-12 };
      bins_[j] = std::move(edges);
    }
  }

  inline int digitize_(double x, const std::vector<double> &edges) const {
    const int B = (int)edges.size() - 1; // bins: 0..B-1, edges: 0..B
    if (B <= 0) return 0;
    if (std::isnan(x)) return B - 1; // route NaN to rightmost

    // bins are [edges[b], edges[b+1]);  find first edge > x, then step back one bin
    auto it = std::upper_bound(edges.begin(), edges.end(), x);
    int idx = int(it - edges.begin()) - 1;
    if (idx < 0)   idx = 0;
    if (idx >= B)  idx = B - 1;
    return idx;
  }

  // --------------------------
  // Tree growth (raw pointers)
  // --------------------------
  std::unique_ptr<Node> grow_regression_(const std::vector<int> &idx, int depth) {
    auto node = std::make_unique<Node>();
    node->n_samples = (int)idx.size();

    // stats
    double s1 = 0.0, s2 = 0.0;
    for (int id : idx) {
      double yv = y_double_[id];
      s1 += yv;
      s2 += yv * yv;
    }
    node->value = idx.empty() ? 0.0 : s1 / idx.size();
    node->impurity = sse_from_sums((double)idx.size(), s1, s2);

    // stop?
    // print depth to output
    if (depth >= max_depth_ || (int)idx.size() < min_samples_split_ ||
        (int)idx.size() < 2 * min_samples_leaf_) {
      return node;
    }

    const SplitResult split = use_histogram_
                                ? best_split_hist_reg_(idx)
                                : best_split_exact_reg_(idx);

    if (!split.valid) return node;

    double decrease = node->impurity - split.cost;
    if (normalize_gain_) decrease /= std::max(1, node->n_samples);
    if (decrease < min_impurity_decrease_) return node;

    node->feature = split.feature;
    node->threshold = split.threshold;

    // Single pass partition to children
    std::vector<int> L; L.reserve(split.left_count);
    std::vector<int> R; R.reserve((int)idx.size() - split.left_count);
    for (int id : idx) {
      double v = X_ptr_[id * n_features_ + split.feature];
      bool go_left = (!std::isnan(v) && v < split.threshold);
      (go_left ? L : R).push_back(id);
    }

    node->left  = grow_regression_(L, depth + 1);
    node->right = grow_regression_(R, depth + 1);
    return node;
  }

  std::unique_ptr<Node> grow_classification_(const std::vector<int> &idx, int depth) {
    auto node = std::make_unique<Node>();
    node->n_samples = (int)idx.size();
    node->class_counts.assign(n_classes_, 0);

    for (int id : idx) node->class_counts[y_int_labels_[id]]++;
    node->impurity = (criterion_ == "gini")
                       ? gini_impurity_from_counts(node->class_counts)
                       : entropy_from_counts(node->class_counts);

    // lowest-index-on-ties
    int best_class = 0;
    int best_count = node->class_counts[0];
    for (int c = 1; c < n_classes_; ++c) {
      if (node->class_counts[c] > best_count) {
        best_class = c;
        best_count = node->class_counts[c];
      }
    }
    node->value = static_cast<double>(best_class);

    if (depth >= max_depth_ || (int)idx.size() < min_samples_split_ ||
        (int)idx.size() < 2 * min_samples_leaf_) {
      return node;
    }

    const SplitResult split = use_histogram_
                                ? best_split_hist_cls_(idx)
                                : best_split_exact_cls_(idx);

    if (!split.valid) return node;

    // parent weighted impurity − child weighted impurity
    const double parent_weighted = node->impurity * node->n_samples;
    double decrease = parent_weighted - split.cost;
    if (normalize_gain_) decrease /= std::max(1, node->n_samples);
    if (decrease < min_impurity_decrease_) return node;

    node->feature = split.feature;
    node->threshold = split.threshold;

    // Single pass partition
    std::vector<int> L; L.reserve(split.left_count);
    std::vector<int> R; R.reserve((int)idx.size() - split.left_count);
    for (int id : idx) {
      double v = X_ptr_[id * n_features_ + split.feature];
      bool go_left = (!std::isnan(v) && v < split.threshold);
      (go_left ? L : R).push_back(id);
    }
    node->left  = grow_classification_(L, depth + 1);
    node->right = grow_classification_(R, depth + 1);
    return node;
  }

  // --------------------------
  // Exact splits (fast)
  // --------------------------
SplitResult best_split_exact_reg_(const std::vector<int> &idx) const {
  const int ns = (int)idx.size();

  double total_s1 = 0.0, total_s2 = 0.0;
  for (int id : idx) {
    const double yv = y_double_[id];
    total_s1 += yv;
    total_s2 += yv * yv;
  }

  SplitResult best;

#pragma omp parallel if (n_features_ >= 4)
  {
    SplitResult thread_best;

#pragma omp for nowait
    for (int j = 0; j < n_features_; ++j) {
      std::vector<std::pair<double, int>> vals;
      vals.reserve(ns);

      double minv =  std::numeric_limits<double>::infinity();
      double maxv = -std::numeric_limits<double>::infinity();

      // --- build vals with finite v only (NaNs skipped) ---
      for (int id : idx) {
        const double v = X_ptr_[id * n_features_ + j];
        if (std::isnan(v)) continue;               // <-- key change
        vals.emplace_back(v, id);
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
      }

      const int ns_eff = (int)vals.size();
      if (ns_eff <= 1 || !(minv < maxv)) continue;

      std::stable_sort(vals.begin(), vals.end(),
                       [](const auto &a, const auto &b){ return a.first < b.first; });

      double left_s1 = 0.0, left_s2 = 0.0;
      int left_n = 0;

      for (int i = 0; i < ns_eff - 1; ++i) {
        const int id_i = vals[i].second;
        const double yv = y_double_[id_i];
        left_s1 += yv; left_s2 += yv * yv; left_n++;

        const int right_n = ns - left_n;  // includes NaNs on the right
        if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;
        if (vals[i].first >= vals[i + 1].first) continue; // skip ties

        const double ls1 = left_s1, ls2 = left_s2;
        const double rs1 = total_s1 - ls1, rs2 = total_s2 - ls2;

        const double cost = sse_from_sums((double)left_n,  ls1, ls2)
                          + sse_from_sums((double)right_n, rs1, rs2);

        const double threshold = 0.5 * (vals[i].first + vals[i + 1].first);
        SplitResult cand{ j, threshold, cost, left_n, true };
        if (is_better(cand, thread_best)) thread_best = cand;
      }
    }

#pragma omp critical
    {
      if (is_better(thread_best, best)) best = thread_best;
    }
  }
  return best;
}
SplitResult best_split_exact_cls_(const std::vector<int> &idx) const {
  const int ns = (int)idx.size();

  // total counts over ALL samples (including those with NaN feature values)
  std::vector<int> total(n_classes_, 0);
  for (int id : idx) total[y_int_labels_[id]]++;

  SplitResult best;

#pragma omp parallel if (n_features_ >= 4)
  {
    SplitResult thread_best;

#pragma omp for nowait
    for (int j = 0; j < n_features_; ++j) {
      std::vector<std::pair<double, int>> vals;
      vals.reserve(ns);

      double minv =  std::numeric_limits<double>::infinity();
      double maxv = -std::numeric_limits<double>::infinity();

      // --- build vals with finite v only (NaNs skipped) ---
      for (int id : idx) {
        const double v = X_ptr_[id * n_features_ + j];
        if (std::isnan(v)) continue;               // <-- key change
        vals.emplace_back(v, id);
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
      }

      const int ns_eff = (int)vals.size();
      if (ns_eff <= 1 || !(minv < maxv)) continue;

      std::stable_sort(vals.begin(), vals.end(),
                       [](const auto &a, const auto &b){ return a.first < b.first; });

      std::vector<int> left_counts(n_classes_, 0);
      int left_n = 0;

      for (int i = 0; i < ns_eff - 1; ++i) {
        const int id_i = vals[i].second;
        left_counts[y_int_labels_[id_i]]++;
        left_n++;

        const int right_n = ns - left_n; // includes NaNs on the right
        if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;
        if (vals[i].first >= vals[i + 1].first) continue;

        std::vector<int> right_counts(n_classes_);
        for (int c = 0; c < n_classes_; ++c)
          right_counts[c] = total[c] - left_counts[c];

        const double impL = (criterion_ == "gini")
                              ? gini_impurity_from_counts(left_counts)
                              : entropy_from_counts(left_counts);
        const double impR = (criterion_ == "gini")
                              ? gini_impurity_from_counts(right_counts)
                              : entropy_from_counts(right_counts);

        const double cost = left_n * impL + right_n * impR;

        const double threshold = 0.5 * (vals[i].first + vals[i + 1].first);
        SplitResult cand{ j, threshold, cost, left_n, true };
        if (is_better(cand, thread_best)) thread_best = cand;
      }
    }

#pragma omp critical
    {
      if (is_better(thread_best, best)) best = thread_best;
    }
  }
  return best;
}


  // --------------------------
  // Histogram splits
  // --------------------------
  SplitResult best_split_hist_reg_(const std::vector<int> &idx) const {
    const int ns = (int)idx.size();
    double total_s1 = 0.0, total_s2 = 0.0;
    for (int id : idx) {
      double yv = y_double_[id];
      total_s1 += yv;
      total_s2 += yv * yv;
    }

    SplitResult best;

#pragma omp parallel if (n_features_ >= 4)
    {
      SplitResult thread_best;

#pragma omp for nowait
      for (int j = 0; j < n_features_; ++j) {
        const auto &edges = bins_[j];
        const int B = (int)edges.size() - 1;
        if (B <= 1) continue;

        std::vector<double> s1(B, 0.0), s2(B, 0.0);
        std::vector<double> cnt(B, 0.0);

        for (int id : idx) {
          const double v = X_ptr_[id * n_features_ + j];
          int b = digitize_(v, edges);
          if (b < 0)   b = 0;
          if (b >= B)  b = B - 1;
          const double yv = y_double_[id];
          s1[b] += yv;
          s2[b] += yv * yv;
          cnt[b] += 1.0;
        }

        std::vector<double> cs1(B), cs2(B), csz(B);
        cs1[0] = s1[0];
        cs2[0] = s2[0];
        csz[0] = cnt[0];
        for (int b = 1; b < B; ++b) {
          cs1[b] = cs1[b - 1] + s1[b];
          cs2[b] = cs2[b - 1] + s2[b];
          csz[b] = csz[b - 1] + cnt[b];
        }

        for (int t = 0; t < B - 1; ++t) {
          const double left_n  = csz[t];
          const double right_n = ns - left_n;
          if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;

          const double ls1 = cs1[t], ls2 = cs2[t];
          const double rs1 = total_s1 - ls1, rs2 = total_s2 - ls2;

          const double cost = sse_from_sums(left_n, ls1, ls2)
                            + sse_from_sums(right_n, rs1, rs2);

          double threshold = edges[t + 1]; // split between bin t and t+1 at boundary
          SplitResult cand{j, threshold, cost, (int)left_n, true};
          if (is_better(cand, thread_best)) thread_best = cand;
        }
      }

#pragma omp critical
      {
        if (is_better(thread_best, best)) best = thread_best;
      }
    }
    return best;
  }

  SplitResult best_split_hist_cls_(const std::vector<int> &idx) const {
    const int ns = (int)idx.size();
    SplitResult best;

#pragma omp parallel if (n_features_ >= 4)
    {
      SplitResult thread_best;

#pragma omp for nowait
      for (int j = 0; j < n_features_; ++j) {
        const auto &edges = bins_[j];
        const int B = (int)edges.size() - 1;
        if (B <= 1) continue;

        std::vector<int> counts(B * n_classes_, 0);
        auto at = [&](int b, int c) -> int & { return counts[b * n_classes_ + c]; };

        for (int id : idx) {
          const double v = X_ptr_[id * n_features_ + j];
          int b = digitize_(v, edges);
          if (b < 0)   b = 0;
          if (b >= B)  b = B - 1;
          at(b, y_int_labels_[id]) += 1;
        }

        std::vector<int> cum(B * n_classes_, 0);
        for (int c = 0; c < n_classes_; ++c) cum[c] = counts[c];
        for (int b = 1; b < B; ++b)
          for (int c = 0; c < n_classes_; ++c)
            cum[b * n_classes_ + c] = cum[(b - 1) * n_classes_ + c] + counts[b * n_classes_ + c];

        std::vector<int> total(n_classes_, 0);
        for (int c = 0; c < n_classes_; ++c)
          total[c] = cum[(B - 1) * n_classes_ + c];

        std::vector<int> cumsz(B, 0);
        for (int b = 0; b < B; ++b) {
          int s = 0;
          for (int c = 0; c < n_classes_; ++c) s += counts[b * n_classes_ + c];
          cumsz[b] = (b == 0 ? s : cumsz[b - 1] + s);
        }

        for (int t = 0; t < B - 1; ++t) {
          const int left_n  = cumsz[t];
          const int right_n = ns - left_n;
          if (left_n < min_samples_leaf_ || right_n < min_samples_leaf_) continue;

          std::vector<int> left_counts(n_classes_), right_counts(n_classes_);
          for (int c = 0; c < n_classes_; ++c) {
            left_counts[c]  = cum[t * n_classes_ + c];
            right_counts[c] = total[c] - left_counts[c];
          }

          const double impL = (criterion_ == "gini")
                                ? gini_impurity_from_counts(left_counts)
                                : entropy_from_counts(left_counts);
          const double impR = (criterion_ == "gini")
                                ? gini_impurity_from_counts(right_counts)
                                : entropy_from_counts(right_counts);

          const double cost = left_n * impL + right_n * impR;

          double threshold = edges[t + 1]; // boundary between t and t+1
          SplitResult cand{j, threshold, cost, left_n, true};
          if (is_better(cand, thread_best)) thread_best = cand;
        }
      }

#pragma omp critical
      {
        if (is_better(thread_best, best)) best = thread_best;
      }
    }
    return best;
  }

  // --------------------------
  // Prediction helpers
  // --------------------------
  inline Node *descend_(const double *x, Node *n) const {
    while (!n->is_leaf()) {
      const double v = x[n->feature];
      const bool left = (!std::isnan(v) && v <= n->threshold);
      n = left ? n->left.get() : n->right.get();
    }
    return n;
  }

  inline double predict_single_(const double *x, Node *n) const {
    n = descend_(x, n);
    if (is_classification_) {
      const int cls_idx = (int)n->value;
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

  // raw data pointers
  const double *X_ptr_ = nullptr; // points directly into numpy memory

  // label maps
  std::vector<double> classes_; // original labels (size K)
  int n_classes_ = 0;
  std::vector<int> y_int_labels_;
  std::vector<double> y_double_;

  // global bins per feature (edges)
  std::vector<std::vector<double>> bins_;

  // root
  std::unique_ptr<Node> root_ = nullptr;
};

// =============================
// pybind11 bindings
// =============================
PYBIND11_MODULE(fast_tree, m) {
  m.doc() = "Fast C++ Decision Tree with exact and histogram splitters (optimized)";

  py::class_<FastDecisionTree>(m, "FastDecisionTree")
      .def(py::init<const std::string &, int, int, int, double, bool, bool, int, const std::string &>(),
           py::arg("criterion") = "gini",
           py::arg("max_depth") = -1,
           py::arg("min_samples_split") = 0,
           py::arg("min_samples_leaf") = 0,
           py::arg("min_impurity_decrease") = 0.0,
           py::arg("normalize_gain") = false, // usually matches sklearn semantics better
           py::arg("use_histogram") = false,
           py::arg("n_bins") = 32,
           py::arg("binning") = "quantile")
      .def("fit", &FastDecisionTree::fit, "Fit the tree")
      .def("predict", &FastDecisionTree::predict, "Predict labels or values")
      .def("predict_proba", &FastDecisionTree::predict_proba, "Predict class probabilities");

  m.def("gini_impurity_from_counts", &gini_impurity_from_counts);
  m.def("entropy_from_counts", &entropy_from_counts);
}
