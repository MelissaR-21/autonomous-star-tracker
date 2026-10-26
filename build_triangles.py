import itertools
import math
from extract_stars import extract_stars

def euclidean_distance(p1, p2):
    """Calculates distance between two point tuples (x1, y1) and (x2, y2) """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def generate_triangles(stars, max_stars_to_triangulate = 30, min_side_px=5.0):
    """
    Generates all unique 3 star combinations and computes invariant feature ratios. 

    Parameters:
        stars: List of tuples (x, y, brightness)
        max_stars_to_triangulate: Limits input star count to prevent combinatorial explosion. 
        min_side_px: Filters out tiny triangles that are likely noise or artifacts.

    Returns:
        List of dictionaries containing indicies, side lengths, and ratio tuples (a/c, b/c). 
    """

    #focus on top n brightest stars 
    tri_stars = stars[:max_stars_to_triangulate]
    num_stars = len(tri_stars)

    triangles = []

    #iterate over unique index triplets
    for i, j, k in itertools.combinations(range(num_stars), 3):
        p1, p2, p3 = tri_stars[i], tri_stars[j], tri_stars[k]

        #calc 3 side lengths
        d12 = euclidean_distance(p1, p2)
        d23 = euclidean_distance(p2, p3)
        d31 = euclidean_distance(p3, p1)

        #sort sides such that a <= b <= c
        sides = sorted ([d12, d23, d31])
        a, b, c = sides[0], sides[1], sides[2]

        #filter out smaller triangles
        if a < min_side_px or c == 0:
            continue

        #scale and rotate invariant ratio pair
        ratio_a_c = a/c
        ratio_b_c = b/c

        triangles.append({
            "star_indicies": (i,j,k),
            "sides": (a, b, c),
            "invariants": (ratio_a_c, ratio_b_c), 
            "coords": ((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]))
        })
    return triangles

if __name__ =="__main__":
    TEST_IMAGE = "Star1_W1.jpg"

    print(f"Extracting star centroids from {TEST_IMAGE}...")
    stars, _ = extract_stars(TEST_IMAGE, sigma_multiplier= 3.0, max_stars= 50)

    print(f"Generating invariant triangles from top 30 stars...")
    triangles = generate_triangles(stars, max_stars_to_triangulate = 30)

    print(f"Generated {len(triangles)} unique star triangles. \n")
    print(f"Sample Triangle Freature Invariants: ")
    for idx, tri in enumerate(triangles[5:10], 1):
        i, j, k = tri["star_indicies"]
        r1, r2 = tri["invariants"]
        a, b, c = tri["sides"]
        print(f"    Triangle {idx}:")
        print(f"        Star Indices: {i}, {j}, {k}")
        print(f"        Side Lengths: a={a:.2f}, b={b:.2f}, c={c:.2f}")
        print(f"        Invariant Tuple: a/c={r1:.3f}, b/c={r2:.3f}")
    





