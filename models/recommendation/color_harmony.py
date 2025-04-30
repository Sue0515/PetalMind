from typing import Tuple, Dict, Any 

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r, g, b = r/255.0, g/255.0, b/255.0  

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val 

    if delta == 0:
        h = 0 
    elif max_val == r: 
        h = ((g - b) / delta) % 6 
    elif max_val == g:
        h = ((b - r) / delta) + 2 
    else:
        h = ((r - g) / delta) + 4 
    
    h = round(h * 60)
    if h < 0:
        h += 360
    
    # 채도(S) 계산
    if max_val == 0:
        s = 0
    else:
        s = delta / max_val
    
    # 명도(V) 계산
    v = max_val
    
    return h, s, v

COLOR_MAP = {
    "red": (255, 0, 0),
    "pink": (255, 192, 203),
    "coral": (255, 127, 80),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "navy": (0, 0, 128),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "brown": (165, 42, 42),
    "white": (255, 255, 255),
    "cream": (255, 253, 208),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "peach": (255, 218, 185),
    "burgundy": (128, 0, 32),
    "maroon": (128, 0, 0),
    "gold": (255, 215, 0)
}

def get_color_rgb(color_name: str) -> Tuple[int, int, int]:
    return COLOR_MAP.get(color_name.lower(), (128, 128, 128)) 

def is_harmonious_color(color1: str, color2: str) -> float:
    """
    Args:
        color1, color2: 색상 이름
        
    Returns:
        harmony_score: 0.0-1.0 범위의 조화 점수
    """
    rgb1 = get_color_rgb(color1)
    rgb2 = get_color_rgb(color2)
    
    h1, s1, v1 = rgb_to_hsv(*rgb1)
    h2, s2, v2 = rgb_to_hsv(*rgb2)
    
    harmony_score = 0.0

    if abs(h1 - h2) < 15: # 같은 색상 
        harmony_score = max(harmony_score, 0.9)

    h_diff = abs((h1 - h2 + 180) % 360 - 180) # 보색 관계 
    if h_diff < 15: 
        harmony_score = max(harmony_score, 0.85)

    if h_diff < 30: # 유사색 관계 
        harmony_score = max(harmony_score, 0,8)

    if abs(h_diff - 120) < 15: # 삼각 배색 
        harmony_score = max(harmony_score, 0.75)

    if abs(h_diff - 90) < 15: # 사각 배색 
        harmony_score = max(harmony_score, 0.7)
    
    if s1 < 0.15 or s2 < 0.15: 
        harmony_score = max(harmony_score, 0.6) # 무채색은 모듣ㄴ 색과 어느정도 조화를 이룸 

    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)

    contrast_score = 0.0

    if 0.3 <= v_diff <= 0.7: # 명도 대비 - 0.3~0.7 차이가 가장 보기 좋음  
        contrast_score = max(contrast_score, 0.8)

    # 채도 대비 (한쪽은 높고 한쪽은 낮은 채도가 좋음)
    if (s1 > 0.7 and s2 < 0.3) or (s1 < 0.3 and s2 > 0.7):
        contrast_score = max(contrast_score, 0.7)

    final_score = harmony_score * 0.7 + contrast_score * 0.3

    return max(final_score, 0.3)