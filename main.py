"""
차량 거리 측정 및 경고 시스템
Vehicle Distance Detection and Warning System

본 시스템은 YOLO 객체 검출과 컴퓨터 비전 기술을 활용하여
전방 차량의 거리를 실시간으로 측정하고 안전 경고를 제공합니다.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import pyttsx3
import threading
import queue
import time
import platform

# ==================== 시스템 설정 ====================
VIDEO_PATH = "./videos/nD_1.mp4"  # 입력 비디오 경로
IMAGE_WIDTH = 640  # 처리할 이미지 너비
IMAGE_HEIGHT = 480  # 처리할 이미지 높이
DETECTION_CONFIDENCE = 0.5  # YOLO 검출 신뢰도 임계값
YOLO_MODEL = 'yolov8s.pt'  # 사용할 YOLO 모델

# ==================== 카메라 캘리브레이션 파라미터 ====================
CAMERA_HEIGHT = 2.0  # 카메라 설치 높이 (미터)
CAMERA_TILT_ANGLE = 15  # 카메라 하향 각도 (도)
CAMERA_FOV = 75  # 카메라 시야각 (도)
ORIGINAL_WIDTH = 1920  # 원본 해상도 너비
ORIGINAL_HEIGHT = 1080  # 원본 해상도 높이

# ==================== 안전 거리 임계값 ====================
WARNING_DISTANCE = 7.0  # 위험 거리 (미터) - 빨간색 경고
CAUTION_DISTANCE = 15.0  # 주의 거리 (미터) - 주황색 경고

# ==================== 차선 검출 설정 ====================
LANE_DETECTION = True  # 차선 검출 활성화 여부
BOTTOM_CROP_RATIO = 0.3  # 하단 영역 제거 비율 (본네트 제거)

# ==================== 이미지 전처리 설정 ====================
ENABLE_HISTOGRAM_EQUALIZATION = True  # 히스토그램 평활화 활성화
HISTOGRAM_METHOD = "CLAHE"  # 평활화 방법 (CLAHE/GLOBAL)
CLAHE_CLIP_LIMIT = 2.0  # CLAHE 클립 한계값
CLAHE_TILE_SIZE = 8  # CLAHE 타일 크기

# ==================== TTS 설정 ====================
TTS_RATE = 160  # 음성 출력 속도 (단어/분)
TTS_VOLUME = 0.9  # 음성 볼륨 (0.0~1.0)
WARNING_COOLDOWN = 4.0  # 음성 경고 간격 (초)

# 차량 클래스 정의
VEHICLE_CLASSES = {
    2: 'car',  # 승용차
    3: 'motorcycle',  # 오토바이
    5: 'bus',  # 버스
    7: 'truck',  # 트럭
    1: 'bicycle'  # 자전거
}


class ImagePreprocessor:
    """
    이미지 전처리 클래스
    히스토그램 평활화를 통한 이미지 품질 향상
    """

    def __init__(self):
        """전처리기 초기화"""
        self.enable_histogram_eq = ENABLE_HISTOGRAM_EQUALIZATION
        self.method = HISTOGRAM_METHOD

        # CLAHE(Contrast Limited Adaptive Histogram Equalization) 객체 생성
        if self.method == "CLAHE":
            self.clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
            )
            print(f"✅ CLAHE 초기화 완료")
        else:
            print(f"✅ Global Histogram Equalization 활성화")

    def apply_histogram_equalization(self, image):
        """
        히스토그램 평활화 적용
        Args:
            image: 입력 이미지 (BGR)
        Returns:
            enhanced_image: 향상된 이미지
        """
        if not self.enable_histogram_eq:
            return image

        try:
            # BGR을 YUV 색공간으로 변환 (Y채널에만 평활화 적용)
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            if self.method == "CLAHE":
                # 적응형 히스토그램 평활화 적용
                yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
            else:
                # 전역 히스토그램 평활화 적용
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

            # YUV를 BGR로 재변환
            enhanced_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return enhanced_image

        except Exception as e:
            print(f"❌ 히스토그램 평활화 오류: {e}")
            return image

    def preprocess_frame(self, frame):
        """
        전체 전처리 파이프라인 실행
        Args:
            frame: 입력 프레임
        Returns:
            processed_frame: 전처리된 프레임
        """
        processed_frame = self.apply_histogram_equalization(frame)
        return processed_frame


class SafeSpeaker:
    """
    안전한 TTS(Text-to-Speech) 시스템
    별도 스레드에서 음성 출력을 처리하여 메인 프로그램 블로킹 방지
    """

    def __init__(self):
        """TTS 시스템 초기화"""
        self.queue = queue.Queue()  # 음성 메시지 큐
        self.engine = None  # TTS 엔진
        self.thread = None  # 백그라운드 스레드
        self.running = True  # 실행 상태
        self.tts_available = False  # TTS 사용 가능 여부
        self._init_engine()

    def _init_engine(self):
        """TTS 엔진 초기화 및 설정"""
        try:
            print("🔊 TTS 엔진 초기화 시작...")

            # pyttsx3 TTS 엔진 생성
            self.engine = pyttsx3.init()

            if self.engine is None:
                print("❌ pyttsx3 엔진 생성 실패")
                return

            # TTS 속성 설정
            self.engine.setProperty('rate', TTS_RATE)
            self.engine.setProperty('volume', TTS_VOLUME)

            # 한국어 음성 찾기 및 설정
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in
                           ['korean', 'ko', '한국', 'heami']):
                        self.engine.setProperty('voice', voice.id)
                        print(f"🇰🇷 한국어 음성 설정: {voice.name}")
                        break

            # TTS 기능 테스트
            test_message = "TTS 테스트"
            self.engine.say(test_message)
            self.engine.runAndWait()

            self.tts_available = True

            # 백그라운드 스레드 시작
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
            print("✅ TTS 시스템 초기화 완료")

        except Exception as e:
            print(f"❌ TTS 초기화 실패: {e}")
            self.tts_available = False

    def speak(self, message: str):
        """
        음성 메시지 출력 요청
        Args:
            message: 출력할 텍스트 메시지
        """
        if not self.tts_available:
            print(f"❌ TTS 비활성화: {message}")
            return

        try:
            print(f"🔊 TTS 큐에 추가: {message}")
            self.queue.put(message)
        except Exception as e:
            print(f"❌ TTS 큐 추가 오류: {e}")

    def _run(self):
        """TTS 백그라운드 스레드 실행 함수"""
        print("🎵 TTS 스레드 시작")

        while self.running:
            try:
                # 1초 타임아웃으로 메시지 대기
                message = self.queue.get(timeout=1)

                if self.engine is not None:
                    print(f"🎵 TTS 재생: {message}")
                    self.engine.say(message)
                    self.engine.runAndWait()

            except queue.Empty:
                continue  # 타임아웃은 정상 상황
            except Exception as e:
                print(f"❌ TTS 재생 오류: {e}")
                time.sleep(0.5)

    def stop(self):
        """TTS 시스템 종료"""
        print("🔚 TTS 시스템 종료 중...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


class LaneDetector:
    """
    차선 검출 클래스
    허프 변환을 이용한 차선 경계선 검출
    """

    def __init__(self):
        """차선 검출기 초기화"""
        self.lane_history = []  # 차선 히스토리 (안정화용)

    def detect_my_lane_boundaries(self, image):
        """
        자차 차선의 양쪽 경계선 검출
        Args:
            image: 입력 이미지
        Returns:
            lane_image: 차선이 그려진 이미지
            left_lane: 왼쪽 차선 좌표
            right_lane: 오른쪽 차선 좌표
            masked_edges: 엣지 검출 결과
        """
        try:
            height, width = image.shape[:2]

            # 1. 그레이스케일 변환 및 노이즈 제거
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 2. Canny 엣지 검출
            edges = cv2.Canny(blur, 130, 160)

            # 3. 관심 영역(ROI) 설정 - 사다리꼴 형태
            mask = np.zeros_like(edges)
            roi_vertices = np.array([[
                (width * 0.25, height * (1 - BOTTOM_CROP_RATIO)),  # 왼쪽 하단
                (width * 0.40, height * 0.45),  # 왼쪽 상단
                (width * 0.60, height * 0.45),  # 오른쪽 상단
                (width * 0.95, height * (1 - BOTTOM_CROP_RATIO))  # 오른쪽 하단
            ]], dtype=np.int32)

            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)

            # 4. 허프 변환을 이용한 직선 검출
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,  # 거리 해상도
                theta=np.pi / 180,  # 각도 해상도
                threshold=40,  # 임계값
                minLineLength=60,  # 최소 선분 길이
                maxLineGap=20  # 최대 선분 간격
            )

            # 5. 검출된 직선을 왼쪽/오른쪽 차선으로 분류
            left_lane, right_lane = self._classify_my_lane_boundaries(lines, width, height)

            # 6. 차선을 이미지에 그리기
            lane_image = self._draw_my_lane_boundaries(image, left_lane, right_lane)

            return lane_image, left_lane, right_lane, masked_edges

        except Exception as e:
            print(f"차선 검출 오류: {e}")
            return image, None, None, None

    def _classify_my_lane_boundaries(self, lines, width, height):
        """
        검출된 직선들을 왼쪽/오른쪽 차선으로 분류
        Args:
            lines: 허프 변환으로 검출된 직선들
            width: 이미지 너비
            height: 이미지 높이
        Returns:
            left_lane: 왼쪽 차선 좌표
            right_lane: 오른쪽 차선 좌표
        """
        if lines is None:
            return None, None

        left_lines = []
        right_lines = []
        img_center = width // 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 기울기 계산 및 필터링
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # 수평선이나 너무 가파른 선 제거
            if abs(slope) < 0.3 or abs(slope) > 2.0:
                continue

            line_center_x = (x1 + x2) // 2

            # 기울기와 위치로 차선 분류
            if slope < -0.3 and line_center_x < img_center + width * 0.1:
                left_lines.append([x1, y1, x2, y2, slope])
            elif slope > 0.3 and line_center_x > img_center - width * 0.1:
                right_lines.append([x1, y1, x2, y2, slope])

        # 각 차선의 대표선 계산
        left_lane = self._extrapolate_lane_boundary(left_lines, height)
        right_lane = self._extrapolate_lane_boundary(right_lines, height)

        return left_lane, right_lane

    def _extrapolate_lane_boundary(self, lines, height):
        """
        여러 선분을 하나의 대표 차선으로 확장
        Args:
            lines: 차선 후보 선분들
            height: 이미지 높이
        Returns:
            lane: 확장된 차선 좌표 [x1, y1, x2, y2]
        """
        if not lines:
            return None

        # 모든 선분의 점들을 수집
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line[:4]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if len(x_coords) < 2:
            return None

        try:
            # 1차 다항식 피팅
            coeffs = np.polyfit(y_coords, x_coords, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # 이미지 전체 높이로 확장
            y1 = int(height * (1 - BOTTOM_CROP_RATIO))  # 하단
            y2 = int(height * 0.45)  # 상단

            x1 = int(slope * y1 + intercept)
            x2 = int(slope * y2 + intercept)

            # 유효성 검사
            if 0 <= x1 <= height * 2 and 0 <= x2 <= height * 2:
                return [x1, y1, x2, y2]
            return None

        except Exception:
            return None

    def _draw_my_lane_boundaries(self, image, left_lane, right_lane):
        """
        차선 경계선을 이미지에 그리기
        Args:
            image: 원본 이미지
            left_lane: 왼쪽 차선 좌표
            right_lane: 오른쪽 차선 좌표
        Returns:
            lane_image: 차선이 그려진 이미지
        """
        lane_image = image.copy()

        # 왼쪽 차선 (빨간색)
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 오른쪽 차선 (파란색)
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        return lane_image

    def is_point_in_my_lane(self, point, left_lane, right_lane, image_width):
        """
        특정 점이 자차 차선 내부에 있는지 판단
        Args:
            point: 확인할 점의 좌표 (x, y)
            left_lane: 왼쪽 차선 좌표
            right_lane: 오른쪽 차선 좌표
            image_width: 이미지 너비
        Returns:
            bool: 차선 내부 여부
        """
        px, py = point

        # 차선 검출 실패 시 기본 영역 사용
        if left_lane is None and right_lane is None:
            return (image_width * 0.2 <= px <= image_width * 0.8) and \
                (py < image_width * (1 - BOTTOM_CROP_RATIO))

        def get_x_at_y(lane, y):
            """주어진 Y좌표에서 차선의 X좌표 계산"""
            if lane is None:
                return None
            x1, y1, x2, y2 = lane
            if y2 == y1:
                return x1
            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            return x

        try:
            left_x = get_x_at_y(left_lane, py)
            right_x = get_x_at_y(right_lane, py)

            # 각 경우에 따른 판단
            if left_x is not None and right_x is not None:
                return (left_x - 10 <= px <= right_x + 10) and \
                    (py < image_width * (1 - BOTTOM_CROP_RATIO))
            elif left_x is not None:
                return (px > left_x - 10) and \
                    (py < image_width * (1 - BOTTOM_CROP_RATIO))
            elif right_x is not None:
                return (px < right_x + 10) and \
                    (py < image_width * (1 - BOTTOM_CROP_RATIO))

            return False

        except Exception:
            return (image_width * 0.2 <= px <= image_width * 0.8) and \
                (py < image_width * (1 - BOTTOM_CROP_RATIO))


class VehicleDistanceDetector:
    """
    차량 거리 측정 메인 클래스
    YOLO 객체 검출과 카메라 캘리브레이션을 이용한 거리 계산
    """

    def __init__(self):
        """시스템 초기화"""
        try:
            print("YOLO 모델 로딩 중...")
            self.model = YOLO(YOLO_MODEL)
            print(f"✅ YOLO 모델 ({YOLO_MODEL}) 로드 완료")
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            raise

        # 검출 설정
        self.detection_confidence = DETECTION_CONFIDENCE

        # 카메라 캘리브레이션 파라미터
        self.camera_height = CAMERA_HEIGHT
        self.camera_angle = CAMERA_TILT_ANGLE
        self.camera_fov = CAMERA_FOV
        self.original_width = ORIGINAL_WIDTH
        self.original_height = ORIGINAL_HEIGHT

        # 안전 거리 임계값
        self.warning_distance = WARNING_DISTANCE
        self.caution_distance = CAUTION_DISTANCE

        # 하위 시스템 초기화
        self.lane_detector = LaneDetector() if LANE_DETECTION else None
        self.speaker = SafeSpeaker()
        self.preprocessor = ImagePreprocessor()

        # 음성 경고 제어
        self.last_warning_time = 0

        # 거리 매핑 테이블 생성
        self.distance_lookup_table = self._precompute_distance_lookup()

        print(f"📷 카메라 설정: 높이 {self.camera_height}m, 각도 {self.camera_angle}°")
        print(f"⚠️ 거리 설정: 경고 {self.warning_distance}m, 주의 {self.caution_distance}m")

    def _precompute_distance_lookup(self):
        """
        카메라 파라미터를 이용한 거리 매핑 테이블 생성
        삼각법을 이용하여 Y좌표별 실제 거리를 미리 계산
        """
        print("📐 거리 매핑 테이블 생성 중...")

        # Y좌표 비율 설정 (상단에서 하단으로)
        y_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        distances = []

        # 카메라 파라미터 계산
        camera_angle_rad = np.radians(self.camera_angle)
        aspect_ratio = self.original_width / self.original_height
        fov_diagonal_rad = np.radians(self.camera_fov)
        fov_v_rad = 2 * np.arctan(np.tan(fov_diagonal_rad / 2) / np.sqrt(1 + aspect_ratio ** 2))

        # 각 Y좌표 비율에 대한 거리 계산
        for y_ratio in y_ratios:
            if y_ratio < 1.0:
                # 삼각법을 이용한 거리 계산
                pixel_y = y_ratio * self.original_height
                center_y = self.original_height / 2
                pixel_offset = pixel_y - center_y

                # 픽셀을 각도로 변환
                pixel_to_angle_ratio = fov_v_rad / self.original_height
                offset_angle = pixel_offset * pixel_to_angle_ratio

                # 지면까지의 각도
                ground_angle = camera_angle_rad + offset_angle

                # 거리 계산
                if ground_angle > 0.01:
                    distance = self.camera_height / np.tan(ground_angle)
                    distance = max(0.5, min(distance, 200.0))  # 범위 제한
                else:
                    distance = 200.0
            else:
                # 최하단은 1m로 고정
                distance = 1.0

            distances.append(distance)

        return {
            'y_ratios': y_ratios,
            'distances': distances
        }

    def calculate_distance_by_interpolation(self, bbox, frame_height):
        """
        보간법을 이용한 거리 계산
        Args:
            bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
            frame_height: 프레임 높이
        Returns:
            distance: 계산된 거리 (미터)
        """
        try:
            x1, y1, x2, y2 = bbox
            bottom_y = y2  # 객체의 하단 Y 좌표

            # 현재 해상도를 원본 해상도로 스케일링
            height_scale = self.original_height / frame_height
            actual_bottom_y = bottom_y * height_scale

            # Y좌표를 비율로 변환
            y_ratio = actual_bottom_y / self.original_height

            # 룩업 테이블에서 선형 보간
            y_ratios = self.distance_lookup_table['y_ratios']
            distances = self.distance_lookup_table['distances']

            if y_ratio <= y_ratios[0]:
                distance = distances[0]
            elif y_ratio >= y_ratios[-1]:
                # 최하단 이상인 경우 외삽
                extra_ratio = y_ratio - 1.0
                distance = 1.0 - (extra_ratio * 2.0)
                distance = max(0.5, distance)
            else:
                # 선형 보간
                distance = np.interp(y_ratio, y_ratios, distances)

            return distance

        except Exception as e:
            print(f"거리 계산 오류: {e}")
            return 10.0

    def speak_warning(self, object_name, distance):
        """
        음성 경고 출력 (쿨다운 적용)
        Args:
            object_name: 객체 이름
            distance: 거리
        """
        try:
            current_time = time.time()
            if current_time - self.last_warning_time > WARNING_COOLDOWN:
                warning_text = f"주의! 전방 {distance:.1f}미터에 {object_name}가 있습니다."
                print(f"🚨 [음성경고] {warning_text}")
                self.speaker.speak(warning_text)
                self.last_warning_time = current_time

        except Exception as e:
            print(f"❌ 음성 경고 오류: {e}")

    def get_distance_color_and_status(self, distance):
        """
        거리에 따른 색상과 상태 반환
        Args:
            distance: 거리 (미터)
        Returns:
            color: BGR 색상 튜플
            status: 상태 문자열
        """
        if distance <= self.warning_distance:
            return (0, 0, 255), "DANGER"  # 빨간색
        elif distance <= self.caution_distance:
            return (0, 165, 255), "CAUTION"  # 주황색
        else:
            return (0, 255, 0), "SAFE"  # 초록색

    def detect_vehicles_with_distance(self, frame):
        """
        차량 검출 및 거리 측정 (자차 차선 필터링 포함)
        Args:
            frame: 입력 프레임
        Returns:
            distance_frame: 결과가 그려진 프레임
            vehicles_with_distance: 검출된 차량 정보 리스트
        """
        try:
            # 1. 차선 경계선 검출
            if self.lane_detector:
                lane_frame, left_lane, right_lane, edges = \
                    self.lane_detector.detect_my_lane_boundaries(frame)
            else:
                lane_frame, left_lane, right_lane, edges = frame, None, None, None

            # 2. YOLO 객체 검출
            results = self.model(
                frame,
                verbose=False,
                conf=self.detection_confidence,
                device='cpu'
            )

            distance_frame = lane_frame.copy()
            vehicles_with_distance = []
            danger_count = 0
            caution_count = 0
            safe_count = 0
            my_lane_count = 0
            other_lane_count = 0

            # 3. 검출된 객체 처리
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # 차량 객체만 처리
                        if class_id in VEHICLE_CLASSES:
                            vehicle_name = VEHICLE_CLASSES[class_id]

                            # 객체의 하단 중앙점 (지면 접촉점)
                            bottom_center = ((x1 + x2) // 2, y2)

                            # 거리 계산
                            distance = self.calculate_distance_by_interpolation(
                                [x1, y1, x2, y2], frame.shape[0])

                            # 자차 차선 내부 확인
                            is_my_lane = self.lane_detector.is_point_in_my_lane(
                                bottom_center, left_lane, right_lane, frame.shape[1])

                            if is_my_lane:
                                my_lane_count += 1

                                # 거리에 따른 색상과 상태 결정
                                color, status = self.get_distance_color_and_status(distance)

                                # 상태별 카운트 증가
                                if status == "DANGER":
                                    danger_count += 1
                                    # 위험 거리에서만 음성 경고
                                    self.speak_warning(vehicle_name, distance)
                                elif status == "CAUTION":
                                    caution_count += 1
                                else:
                                    safe_count += 1

                                # 바운딩 박스 그리기
                                cv2.rectangle(distance_frame, (x1, y1), (x2, y2), color, 4)

                                # 지면 접촉점 표시
                                cv2.circle(distance_frame, bottom_center, 8, color, -1)

                                # 정보 라벨 표시
                                label = f"{vehicle_name}: {distance:.1f}m [{status}]"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                                # 라벨 배경 그리기
                                cv2.rectangle(distance_frame, (x1, y1 - label_size[1] - 10),
                                              (x1 + label_size[0] + 10, y1), color, -1)

                                # 라벨 텍스트 그리기
                                cv2.putText(distance_frame, label, (x1 + 5, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                # 신뢰도 표시
                                cv2.putText(distance_frame, f"{confidence:.2f}", (x2 - 40, y1 + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                                # 검출 정보 저장
                                vehicles_with_distance.append({
                                    'name': vehicle_name,
                                    'distance': distance,
                                    'status': status,
                                    'confidence': confidence,
                                    'my_lane': True
                                })

                            else:
                                other_lane_count += 1
                                # 타차선 차량은 회색 점선으로 표시
                                cv2.rectangle(distance_frame, (x1, y1), (x2, y2),
                                              (128, 128, 128), 2, cv2.LINE_4)

                                # 타차선 라벨
                                other_label = f"{vehicle_name}: {distance:.1f}m [other lane]"
                                cv2.putText(distance_frame, other_label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # 4. 통계 정보 표시
            stats_text = f"My Lane: {my_lane_count} | Other: {other_lane_count} | " \
                         f"Danger: {danger_count} | Caution: {caution_count} | Safe: {safe_count}"
            cv2.putText(distance_frame, stats_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 5. 시스템 상태 표시
            # TTS 상태
            tts_status = "TTS: OK" if self.speaker.tts_available else "TTS: FAIL"
            tts_color = (0, 255, 0) if self.speaker.tts_available else (0, 0, 255)
            cv2.putText(distance_frame, tts_status, (10, distance_frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 2)

            # 차선 검출 상태
            if left_lane and right_lane:
                lane_status = "BOTH BOUNDARIES OK"
                lane_color = (0, 255, 0)
            elif left_lane:
                lane_status = "LEFT BOUNDARY ONLY"
                lane_color = (0, 0, 255)
            elif right_lane:
                lane_status = "RIGHT BOUNDARY ONLY"
                lane_color = (255, 0, 0)
            else:
                lane_status = "NO BOUNDARIES"
                lane_color = (128, 128, 128)

            cv2.putText(distance_frame, f"Lane: {lane_status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 2)

            # 6. 범례 표시
            legend_y = distance_frame.shape[0] - 60

            # 안전 (초록색)
            cv2.rectangle(distance_frame, (10, legend_y - 15), (30, legend_y), (0, 255, 0), -1)
            cv2.putText(distance_frame, f"Safe (>{self.caution_distance}m)", (35, legend_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 주의 (주황색)
            cv2.rectangle(distance_frame, (180, legend_y - 15), (200, legend_y), (0, 165, 255), -1)
            cv2.putText(distance_frame, f"Caution ({self.warning_distance}-{self.caution_distance}m)",
                        (205, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 위험 (빨간색)
            cv2.rectangle(distance_frame, (400, legend_y - 15), (420, legend_y), (0, 0, 255), -1)
            cv2.putText(distance_frame, f"Danger (<{self.warning_distance}m)", (425, legend_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            return distance_frame, vehicles_with_distance

        except Exception as e:
            print(f"❌ 차량 검출 및 거리 측정 오류: {e}")
            return frame, []

    def detect_vehicles(self, frame):
        """
        디버그용 차량 검출 (모든 객체 표시)
        Args:
            frame: 입력 프레임
        Returns:
            debug_frame: 디버그 정보가 표시된 프레임
            vehicle_count: 검출된 차량 수
        """
        try:
            # 이미지 전처리
            processed_frame = self.preprocessor.preprocess_frame(frame)

            # YOLO 검출
            results = self.model(
                processed_frame,
                verbose=False,
                conf=self.detection_confidence,
                device='cpu'
            )

            debug_frame = frame.copy()
            detected_count = 0
            vehicle_count = 0

            # 검출된 모든 객체 표시
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # 클래스 이름 가져오기
                        class_name = self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}"

                        detected_count += 1

                        # 모든 객체를 하늘색으로 표시
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                        # 차량 객체인지 확인
                        if class_id in VEHICLE_CLASSES:
                            vehicle_count += 1

                        # 객체 정보 라벨
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(debug_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 검출 통계 표시
            stats = f"Total Objects: {detected_count} | Vehicles: {vehicle_count}"
            cv2.putText(debug_frame, stats, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            return debug_frame, vehicle_count

        except Exception as e:
            print(f"❌ 디버그 검출 오류: {e}")
            return frame, 0

    def run(self, video_path):
        """
        메인 시스템 실행
        Args:
            video_path: 입력 비디오 파일 경로
        """
        cap = None
        try:
            print(f"비디오 파일 로딩: {video_path}")

            # 비디오 파일 열기 (여러 백엔드 시도)
            cap = cv2.VideoCapture(video_path)

            # 백엔드 대체 시도
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)

            if not cap.isOpened():
                print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
                return

            # 비디오 정보 출력
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 비디오 정보: {width}x{height}, {fps:.1f}fps, {frame_count}프레임")

            print("=== 차량 거리 측정 및 경고 시스템 시작 ===")
            print("조작법: 'q' 종료, 'p' 일시정지/재생, 'r' 재시작, 'h' 전처리 토글")

            frame_count = 0
            paused = False

            # 메인 처리 루프
            while True:
                # 프레임 읽기 (일시정지 상태가 아닐 때만)
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("📹 비디오 재생 완료")
                        break
                    frame_count += 1

                if 'frame' in locals():
                    # 프레임 크기 조정
                    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

                    # 1. 디버그 창 - YOLO 원본 검출 결과
                    debug_frame, vehicle_count = self.detect_vehicles(frame)

                    # 2. 메인 창 - 거리 측정 및 안전 경고
                    distance_frame, vehicles_with_distance = \
                        self.detect_vehicles_with_distance(frame)

                    # 3. 전처리 결과 창 (선택사항)
                    if ENABLE_HISTOGRAM_EQUALIZATION:
                        preprocessed_frame = self.preprocessor.preprocess_frame(frame)
                        cv2.imshow('Image Preprocessing Result', preprocessed_frame)

                    # 프레임 번호 표시
                    frame_info = f"Frame: {frame_count}"
                    cv2.putText(debug_frame, frame_info, (10, debug_frame.shape[0] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(distance_frame, frame_info, (10, distance_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # 일시정지 상태 표시
                    if paused:
                        cv2.putText(debug_frame, "PAUSED", (IMAGE_WIDTH // 2 - 50, IMAGE_HEIGHT // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        cv2.putText(distance_frame, "PAUSED", (IMAGE_WIDTH // 2 - 50, IMAGE_HEIGHT // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    # 결과 창 표시
                    cv2.imshow('Debug - YOLO Detection', debug_frame)
                    cv2.imshow('Main - Distance Measurement & Warning', distance_frame)

                # 키보드 입력 처리
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    print("🔚 시스템 종료")
                    break
                elif key == ord('p') or key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️ 일시정지' if paused else '▶️ 재생'}")
                elif key == ord('r'):
                    # 비디오 재시작
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    paused = False
                    print("🔄 비디오 재시작")
                elif key == ord('h'):
                    # 히스토그램 평활화 토글
                    self.preprocessor.enable_histogram_eq = not self.preprocessor.enable_histogram_eq
                    status = "ON" if self.preprocessor.enable_histogram_eq else "OFF"
                    print(f"🎨 히스토그램 평활화: {status}")

        except Exception as e:
            print(f"❌ 시스템 실행 오류: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 리소스 정리
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'speaker'):
                self.speaker.stop()
            print("✅ 시스템 종료 완료")


def main():
    """
    메인 함수 - 프로그램 진입점
    """
    try:
        print("=== 차량 거리 측정 및 경고 시스템 ===")
        print("개발: AI 기반 실시간 안전 운전 지원 시스템")
        print("기능: YOLO 객체 검출, 거리 계산, 차선 인식, 음성 경고")
        print("=" * 50)

        # 시스템 초기화
        detector = VehicleDistanceDetector()

        # 비디오 파일 확인 및 실행
        if os.path.exists(VIDEO_PATH):
            detector.run(VIDEO_PATH)
        else:
            print(f"❌ 비디오 파일을 찾을 수 없습니다: {VIDEO_PATH}")
            print("현재 디렉토리의 비디오 파일들:")
            for file in os.listdir("."):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    print(f"  - {file}")

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의한 프로그램 중단")
    except Exception as e:
        print(f"❌ 시스템 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
=== 시스템 구조 요약 ===

1. ImagePreprocessor: 이미지 품질 향상 (히스토그램 평활화)
2. SafeSpeaker: 비동기 TTS 음성 경고 시스템
3. LaneDetector: 허프 변환 기반 차선 검출
4. VehicleDistanceDetector: 메인 시스템 (YOLO + 거리 계산)

=== 핵심 알고리즘 ===

1. 객체 검출: YOLOv8을 이용한 실시간 차량 검출
2. 거리 계산: 카메라 캘리브레이션과 삼각법 기반 거리 추정
3. 차선 인식: Canny 엣지 + 허프 변환을 이용한 차선 경계 검출
4. 안전 판단: 자차 차선 내 객체만 필터링하여 안전 거리 경고

=== 기술적 특징 ===

- 실시간 처리 최적화
- 다중 스레드 TTS 시스템
- 적응형 이미지 전처리
- 카메라 파라미터 기반 정확한 거리 측정
- 직관적인 시각적 피드백 (색상 코딩)
"""