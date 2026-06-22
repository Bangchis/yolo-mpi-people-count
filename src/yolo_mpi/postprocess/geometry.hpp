// Box merging, NMS, correctness verification, and CSV output.
// Rank 0 calls these after it has gathered detections from all ranks.

// Compute bbox area, clamping invalid width/height to zero.
static double box_area(const Detection& d) {
    return std::max(0.0, d.x2 - d.x1) * std::max(0.0, d.y2 - d.y1);
}

// Compute bbox width with a zero lower bound.
static double box_width(const Detection& d) {
    return std::max(0.0, d.x2 - d.x1);
}

// Compute bbox height with a zero lower bound.
static double box_height(const Detection& d) {
    return std::max(0.0, d.y2 - d.y1);
}

// Compute the overlapping area between two boxes.
static double intersection_area(const Detection& a, const Detection& b) {
    double ix1 = std::max(a.x1, b.x1);
    double iy1 = std::max(a.y1, b.y1);
    double ix2 = std::min(a.x2, b.x2);
    double iy2 = std::min(a.y2, b.y2);
    double iw = std::max(0.0, ix2 - ix1);
    double ih = std::max(0.0, iy2 - iy1);
    return iw * ih;
}

// Standard intersection-over-union used by global NMS.
static double iou(const Detection& a, const Detection& b) {
    double inter = intersection_area(a, b);
    double area_a = box_area(a);
    double area_b = box_area(b);
    double denom = area_a + area_b - inter;
    return denom > 0 ? inter / denom : 0.0;
}

// Measure how much the smaller box is covered by the other box.
static double intersection_over_smaller(const Detection& a, const Detection& b) {
    double smaller = std::min(box_area(a), box_area(b));
    if (smaller <= 0) return 0.0;
    return intersection_area(a, b) / smaller;
}

// Measure 1D overlap relative to the smaller interval.
static double axis_overlap_ratio(double a1, double a2, double b1, double b2) {
    double overlap = std::max(0.0, std::min(a2, b2) - std::max(a1, b1));
    double smaller = std::min(std::max(0.0, a2 - a1), std::max(0.0, b2 - b1));
    return smaller > 0 ? overlap / smaller : 0.0;
}

// Measure normalized gap between two 1D intervals.
static double axis_gap_ratio(double a1, double a2, double b1, double b2, double scale) {
    double gap = std::max(0.0, std::max(a1, b1) - std::min(a2, b2));
    return scale > 0 ? gap / scale : 0.0;
}
