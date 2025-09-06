import cv2
import numpy as np
import pytesseract
import networkx as nx
import matplotlib.pyplot as plt


pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

def detect_nodes(gray_image):
    # Blur helps detection
    blurred = cv2.medianBlur(gray_image, 5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=20, minRadius=10, maxRadius=60
    )
    nodes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            nodes.append((x, y))
    print(f"[DEBUG] Detected {len(nodes)} nodes")

    # Debug: draw detected circles
    debug_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        for (x, y, r) in circles[0, :]:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected nodes")
    plt.show()

    return nodes


def detect_edges(color_img, nodes, node_radius=50):
    mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
    for (x, y) in nodes:
        cv2.circle(mask, (x, y), node_radius+5, 255, -1)
    img_no_nodes = cv2.inpaint(color_img, mask, 3, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(img_no_nodes, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Debug edges
    plt.imshow(edges, cmap="gray")
    plt.title("Canny edges")
    plt.show()

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=20, maxLineGap=30)
    print(f"[DEBUG] Detected {0 if lines is None else len(lines)} edges")
    return lines if lines is not None else []


def extract_weight(color_img, x, y):
    """
    Extracts the weight number from the circular ROI around (x, y)
    Works for any colored edge/weight, robust to anti-aliasing and small text
    """
    h, w = color_img.shape[:2]

    # 1️⃣ Extract circular ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), 40, 255, -1)
    roi = cv2.bitwise_and(color_img, color_img, mask=mask)
    roi = roi[max(0, y-40):min(h, y+40),
              max(0, x-40):min(w, x+40)]

    if roi.size == 0:
        return None

    # 2️⃣ Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Optional: mask out very bright pixels (white background)
    # Keep pixels that are darker than bright background
    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_TOZERO_INV)

    # 4️⃣ Upsample for small text
    gray = cv2.resize(gray, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 5️⃣ Enhance contrast
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.equalizeHist(gray)

    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title(f"OCR gray patch at ({x},{y})")
    plt.show()

    # 6️⃣ Adaptive threshold for OCR
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    # 7️⃣ Optional: visualize for debugging
    # plt.imshow(thresh, cmap='gray'); plt.title(f"ROI at ({x},{y})"); plt.show()

    # 8️⃣ OCR using Tesseract
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 10 -c tessedit_char_whitelist=0123456789"
    ).strip()

    if text.isdigit():
        print(f"[DEBUG OCR] ROI at ({x},{y}) -> '{text}'")
        return int(text)
    else:
        print(f"[DEBUG OCR] ROI at ({x},{y}) -> OCR failed ('{text}')")
        return None



def find_two_closest_nodes(p1, p2, nodes, threshold=100):
    # returns node indices for line endpoints
    def closest(pt):
        dists = [np.hypot(pt[0]-nx, pt[1]-ny) for nx,ny in nodes]
        idx = np.argmin(dists)
        return idx, dists[idx]
    n1, d1 = closest(p1)
    n2, d2 = closest(p2)
    print(f"[DEBUG] Line endpoints {p1}->{p2} matched to nodes {n1},{n2} with dists {d1:.1f},{d2:.1f}")

    if d1 < threshold and d2 < threshold and n1 != n2:
        return n1, n2
    return None, None


def debug_draw_raw_lines(img, lines):
    dbg = img.copy()
    if lines is not None:
        for (x1,y1,x2,y2) in [l[0] for l in lines]:
            cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 2)  # red lines
    plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
    plt.title("Raw Hough lines")
    plt.show()

def get_edge_color(img, x1, y1, x2, y2, patch_size=7):
    midx, midy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = img.shape[:2]

    # Extract patch around midpoint
    half = patch_size // 2
    x1p, x2p = max(0, midx - half), min(w, midx + half + 1)
    y1p, y2p = max(0, midy - half), min(h, midy + half + 1)
    patch = img[y1p:y2p, x1p:x2p]

    if patch.size == 0:
        return None

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Mask out nearly white pixels (background)
    lower = np.array([0, 30, 30])    # avoid white/gray
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_patch, lower, upper)

    valid = hsv_patch[mask > 0]
    if len(valid) == 0:
        return None

    # Average HSV of valid pixels
    avg_hsv = np.mean(valid, axis=0).astype(int)
    return avg_hsv


# 5. Build graph
def build_graph(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    nodes = detect_nodes(gray)
    lines = detect_edges(color_img, nodes)
    debug_draw_raw_lines(color_img, lines)

    
    edges = []
    for x1,y1,x2,y2 in [l[0] for l in lines]:
        n1, n2 = find_two_closest_nodes((x1,y1),(x2,y2),nodes)

        if n1 is not None and n2 is not None and n1 != n2:
            midx, midy = (x1+x2)//2, (y1+y2)//2
            edge_color = get_edge_color(color_img, x1, y1, x2, y2)
            print("-------------------------",edge_color, "-------------------------")
            w = extract_weight(color_img, midx, midy)
            if w is not None:
                edges.append((n1, n2, w))

    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i, pos=nodes[i])
    for u,v,w in edges:
        G.add_edge(u,v, weight=w)
    return G


def draw_graph_debug(color_img, G):
    img_copy = color_img.copy()

    # Draw nodes
    for node, data in G.nodes(data=True):
        x, y = data['pos']
        cv2.circle(img_copy, (x, y), 25, (0, 255, 0), 2)  # green circle
        cv2.putText(img_copy, str(node), (x-10, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw edges with weights
    for u, v, data in G.edges(data=True):
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        cv2.line(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue line

        # draw weight near the midpoint
        midx, midy = (x1+x2)//2, (y1+y2)//2
        cv2.putText(img_copy, str(data['weight']), (midx, midy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show with matplotlib (so colors look correct)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

img = cv2.imread("graph_0.png")
G = build_graph(img)

print("Nodes:", G.nodes(data=True))
print("Edges:", G.edges(data=True))

draw_graph_debug(img, G)

