import math
import numpy as np
from typing import Tuple, List, Dict

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # 명도
    v = max_val
    
    # 채도
    s = 0 if max_val == 0 else diff / max_val
    
    # 색조
    h = 0
    if diff == 0:
        h = 0
    elif max_val == r:
        h = 60 * ((g - b) / diff % 6)
    elif max_val == g:
        h = 60 * ((b - r) / diff + 2)
    elif max_val == b:
        h = 60 * ((r - g) / diff + 4)
    
    if h < 0:
        h += 360
    
    return (h, s, v)

def color_name_to_rgb(color_name: str) -> Tuple[int, int, int]:
    color_map = {
        'red': (220, 50, 50),
        'pink': (255, 155, 170),
        'orange': (255, 165, 0),
        'yellow': (255, 215, 0),
        'green': (50, 205, 50),
        'blue': (30, 144, 255),
        'purple': (138, 43, 226),
        'violet': (148, 0, 211),
        'lavender': (230, 190, 255),
        'white': (255, 255, 255),
        'cream': (255, 253, 208),
        'ivory': (255, 255, 240),
        'beige': (245, 245, 220),
        'brown': (165, 42, 42),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'coral': (255, 127, 80),
        'peach': (255, 218, 185)
    }
    
    color_name = color_name.lower()

    if color_name in color_map:
        return color_map[color_name]
    
    return (128, 128, 128)

def calculate_color_distance(color1: str, color2: str) -> float:

    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)

    hsv1 = rgb_to_hsv(*rgb1)
    hsv2 = rgb_to_hsv(*rgb2)

    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    h_dist = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180.0
    
    # 채도 거리
    s_dist = abs(s1 - s2)
    
    # 명도 거리
    v_dist = abs(v1 - v2)

    weighted_dist = 0.6 * h_dist + 0.2 * s_dist + 0.2 * v_dist
    
    return weighted_dist

def is_analogous(color1: str, color2: str) -> bool:
   
    # 두 색상이 유사색 관계인지 확인
    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)

    h1 = rgb_to_hsv(*rgb1)[0]
    h2 = rgb_to_hsv(*rgb2)[0]
    
    h_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))

    return h_diff <= 45

def is_complementary(color1: str, color2: str) -> bool:

    # 두 색상이 보색 관계인지 확인

    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)

    h1 = rgb_to_hsv(*rgb1)[0]
    h2 = rgb_to_hsv(*rgb2)[0]

    h_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))

    return 150 <= h_diff <= 210

def is_triadic(color1: str, color2: str, color3: str) -> bool:

    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)
    rgb3 = color_name_to_rgb(color3)

    h1 = rgb_to_hsv(*rgb1)[0]
    h2 = rgb_to_hsv(*rgb2)[0]
    h3 = rgb_to_hsv(*rgb3)[0]

    h_diff12 = min(abs(h1 - h2), 360 - abs(h1 - h2))
    h_diff23 = min(abs(h2 - h3), 360 - abs(h2 - h3))
    h_diff31 = min(abs(h3 - h1), 360 - abs(h3 - h1))

    return (90 <= h_diff12 <= 150 and 
            90 <= h_diff23 <= 150 and 
            90 <= h_diff31 <= 150)


def is_neutral(color: str) -> bool:

    neutral_colors = ['white', 'cream', 'ivory', 'beige', 'gray', 'black']

    if color.lower() in neutral_colors:
        return True

    rgb = color_name_to_rgb(color)

    _, s, _ = rgb_to_hsv(*rgb)

    return s < 0.2


def is_harmonious_color(color1: str, color2: str) -> float:

    # 두 색상이 조화로운지 평가

    if is_neutral(color1) or is_neutral(color2):
        return 0.8

    if is_analogous(color1, color2):
        return 0.9

    if is_complementary(color1, color2):
        return 0.7

    color_dist = calculate_color_distance(color1, color2)

    harmony_score = 1.0 - color_dist
    
    return max(0.2, harmony_score) 


def evaluate_flower_harmony(main_flower: Dict, medium_flowers: List[Dict], 
                            small_flowers: List[Dict]) -> float:
    
    main_color = main_flower['color'].lower()
    harmony_scores = []
    
    # 메인 꽃과 중형 꽃 사이의 조화 평가
    for flower in medium_flowers:
        flower_color = flower['color'].lower()
        score = is_harmonious_color(main_color, flower_color)
        harmony_scores.append(score)
    
    # 메인 꽃과 소형 꽃 사이의 조화 평가
    for flower in small_flowers:
        flower_color = flower['color'].lower()
        score = is_harmonious_color(main_color, flower_color)
        harmony_scores.append(score)
    
    # 모든 꽃 색상 목록
    all_colors = [main_color] + [f['color'].lower() for f in medium_flowers + small_flowers]
    
    has_triadic = False
    if len(all_colors) >= 3:
        for i in range(len(all_colors)):
            for j in range(i+1, len(all_colors)):
                for k in range(j+1, len(all_colors)):
                    if is_triadic(all_colors[i], all_colors[j], all_colors[k]):
                        has_triadic = True
                        break
    
    # 평균 조화 점수 계산
    avg_harmony = sum(harmony_scores) / len(harmony_scores) if harmony_scores else 0.5
    
    if has_triadic:
        avg_harmony = min(1.0, avg_harmony + 0.1)
    
    return avg_harmony