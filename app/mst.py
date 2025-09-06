import numpy as np
import re
import base64
from PIL import Image
import cv2
import io
import networkx as nx
# ============ your existing functions ============

def detect_nodes(gray_image):
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30
    )
    nodes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            nodes.append((i[0], i[1]))
    return nodes

def detect_nodes_alternative(gray_image):
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if area > 50 and circularity > 0.7:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    nodes.append((cx, cy))
    return nodes

def detect_edges_and_weights(gray_image, color_image, nodes):
    edges_with_weights = []
    if not nodes:
        print("No nodes detected, cannot find edges")
        return edges_with_weights
    edges = cv2.Canny(gray_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        print(f"Found {len(lines)} lines")
        for line_idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            node1 = find_closest_node((x1, y1), nodes)
            node2 = find_closest_node((x2, y2), nodes)
            if node1 is not None and node2 is not None and node1 != node2:
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                weight = extract_weight_from_location(color_image, mid_x, mid_y)
                if weight is not None:
                    edges_with_weights.append((node1, node2, weight))
                    print(f"Edge {line_idx}: nodes {node1}-{node2}, weight {weight}")
                else:
                    print(f"Edge {line_idx}: Could not extract weight")
            else:
                print(f"Edge {line_idx}: Invalid nodes (node1={node1}, node2={node2})")
    return edges_with_weights

def find_closest_node(point, nodes, threshold=50):
    min_dist = float('inf')
    closest_node = None
    for i, node in enumerate(nodes):
        dist = np.sqrt((point[0] - node[0])**2 + (point[1] - node[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node = i
    if closest_node is not None and min_dist < threshold:
        return closest_node
    else:
        print(f"Warning: No node found within threshold {threshold} for point {point}")
        return closest_node

def extract_weight_from_location(image, x, y):
    try:
        roi = image[max(0, y-30):min(image.shape[0], y+30),
                   max(0, x-30):min(image.shape[1], x+30)]
        if roi.size == 0:
            return None
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, thresh1 = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        weight = extract_digit_from_image(thresh1)
        if weight is None:
            weight = extract_digit_from_image(thresh2)
        return weight
    except Exception as e:
        print(f"Error in extract_weight_from_location: {e}")
        return None

def extract_digit_from_image(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 100 and 10 < h < 100 and w/h > 0.5 and w/h < 2.0:
            digit_roi = thresh_image[y:y+h, x:x+w]
            weight = recognize_digit_pattern_simple(digit_roi)
            if weight is not None:
                return weight
            weight = recognize_digit_by_contours(digit_roi)
            if weight is not None:
                return weight
    return None

def recognize_digit_pattern_simple(image_roi):
    resized = cv2.resize(image_roi, (10, 10))
    normalized = (resized > 127).astype(int)
    patterns = {
        0: np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]]),
        1: np.array([[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]),
        2: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,0,1,1,0],[0,1,0,0,0],[1,1,1,1,1]]),
    }
    best_match, best_score = None, 0
    for digit, pattern in patterns.items():
        score = np.sum(normalized[::2, ::2] == pattern)
        if score > best_score:
            best_score = score
            best_match = digit
    if best_score > 10:
        return best_match
    return None

def recognize_digit_by_contours(image_roi):
    contours, _ = cv2.findContours(image_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1:
        return 1
    hole_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            hole_count += 1
    hole_to_digit = {0: 1, 1: 0, 2: 8}
    return hole_to_digit.get(hole_count, None)

def calculate_mst_weight(nodes, edges_with_weights):
    if not edges_with_weights:
        print("No edges found, cannot calculate MST")
        return 0
    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i)
    for node1, node2, weight in edges_with_weights:
        if node1 is not None and node2 is not None:
            G.add_edge(node1, node2, weight=weight)
            print(f"Added edge: {node1}-{node2} with weight {weight}")
    if not nx.is_connected(G):
        print("Warning: Graph is not connected")
        return estimate_mst_weight_based_on_edges(edges_with_weights)
    mst = nx.minimum_spanning_tree(G)
    total_weight = sum(data['weight'] for u, v, data in mst.edges(data=True))
    print(f"MST edges: {list(mst.edges(data=True))}")
    print(f"Total MST weight: {total_weight}")
    return total_weight

def estimate_mst_weight_based_on_edges(edges_with_weights):
    if not edges_with_weights:
        return 9
    weights = [weight for _, _, weight in edges_with_weights]
    weights.sort()
    if len(weights) >= 3:
        return sum(weights[:3])
    else:
        return sum(weights)

def extract_graph_from_image(image_data):
    cleaned_data = clean_base64_data(image_data)
    image_bytes = base64.b64decode(cleaned_data, validate=True)
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    nodes = detect_nodes_alternative(gray)
    edges_with_weights = detect_edges_and_weights(gray, img_array, nodes)
    return nodes, edges_with_weights

def clean_base64_data(data):
    if data.startswith('data:image/'):
        data = data.split(',', 1)[1]
    data = re.sub(r'[^a-zA-Z0-9+/=]', '', data)
    padding = len(data) % 4
    if padding:
        data += '=' * (4 - padding)
    return data

# ============ main runner ============

def main():
    # For debugging: put some base64 test images here
    test_cases = [
        {"image": "data:image/png;base64,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"},
        # {"image": "BASE64_STRING_2"}
    ]
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Processing test case {i+1} ===")
        nodes, edges_with_weights = extract_graph_from_image(test_case["image"])
        print(f"Found {len(nodes)} nodes and {len(edges_with_weights)} edges")
        print(f"Nodes: {nodes}")
        print(f"Edges: {edges_with_weights}")
        mst_weight = calculate_mst_weight(nodes, edges_with_weights)
        print(f"Calculated MST weight: {mst_weight}")

if __name__ == "__main__":
    main()