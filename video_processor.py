from moviepy.editor import *
import os

# 입력 영상과 자막 (한글)
card_videos = [
    "/content/drive/MyDrive/test용 파일 (20250721)/영어 sample-1 (sound 없음).mp4",
    "/content/drive/MyDrive/test용 파일 (20250721)/영어 sample-2 (sound 없음).mp4",
    "/content/drive/MyDrive/test용 파일 (20250721)/영어 sample-3 (sound 없음).mp4",
    "/content/drive/MyDrive/test용 파일 (20250721)/영어 sample-4 (sound 없음).mp4",
]

english_captions = [
    "/content/drive/MyDrive/test용 파일 (20250721)/en-mp3/01_The_surgeon_resects_the_bile_duct_cyst_.mp3",
    "/content/drive/MyDrive/test용 파일 (20250721)/en-mp3/02_then_resects_the_small_intestine_.mp3",
    "/content/drive/MyDrive/test용 파일 (20250721)/en-mp3/03_After_that_the_resected_small_intestine_is_anasto.mp3",
    "/content/drive/MyDrive/test용 파일 (20250721)/en-mp3/04_and_the_small_intestine_is_anastomosed_to_the_smal.mp3",
]

# 영어 텍스트 (MP3 파일명에서 추출)
english_texts = [
    "The surgeon resects the bile duct cyst",
    "then resects the small intestine",
    "After that the resected small intestine is anastomosed",
    "and the small intestine is anastomosed to the small"
]

def split_text_by_bytes(text, max_bytes=10):
    """텍스트를 바이트 길이 기준으로 분할"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if len(test_line.encode('utf-8')) <= max_bytes:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # 단어 자체가 max_bytes를 초과하는 경우
                lines.append(word)
                current_line = ""
    
    if current_line:
        lines.append(current_line)
    
    return lines

def create_text_clip(text, duration, video_size):
    """텍스트 클립 생성 (자동 줄바꿈 및 가운데 정렬)"""
    lines = split_text_by_bytes(text, max_bytes=10)
    
    if len(lines) == 1:
        # 한 줄인 경우
        txt_clip = TextClip(
            lines[0],
            fontsize=50,
            color='white',
            font='Arial-Bold'
        ).set_duration(duration).set_position('center')
    else:
        # 두 줄 이상인 경우
        line_clips = []
        for i, line in enumerate(lines):
            line_clip = TextClip(
                line,
                fontsize=50,
                color='white',
                font='Arial-Bold'
            ).set_duration(duration)
            line_clips.append(line_clip)
        
        # 각 줄을 세로로 배치 (가운데 정렬)
        total_height = sum([clip.h for clip in line_clips]) + (len(line_clips) - 1) * 10  # 10px 간격
        start_y = (video_size[1] - total_height) // 2
        
        positioned_clips = []
        current_y = start_y
        for clip in line_clips:
            positioned_clip = clip.set_position(('center', current_y))
            positioned_clips.append(positioned_clip)
            current_y += clip.h + 10
        
        txt_clip = CompositeVideoClip(positioned_clips, size=video_size)
    
    return txt_clip

clips = []
padding = 0.5
audio_delay = 0.5

#to_ImageClip(...)이 내부적으로 축소/재샘플링을 하거나 fps/크기 정보가 애매해서
#원본 프레임을 그대로 써서 해상도 유지하려면
#get_frame으로 픽셀 데이터를 직접 뽑아서 ImageClip을 만들고, 원본 크기·fps를 명시해주는 게 깔끔
for idx, (video_path, eng_path, eng_text) in enumerate(zip(card_videos, english_captions, english_texts)):
    print(f"Processing {video_path}...")

    # 1. MP3 길이 계산
    original_audio = AudioFileClip(eng_path)
    delayed_audio = concatenate_audioclips([
        AudioClip(lambda t: 0, duration=audio_delay),  # 무음 추가
        original_audio
    ])
    mp3_duration = delayed_audio.duration

    # 2. 영상 불러오기
    base_clip = VideoFileClip(video_path)
    if base_clip.duration - mp3_duration <= padding:
        adding = padding - (base_clip.duration - mp3_duration)
        if base_clip.duration < mp3_duration:
            # 원본 해상도/품질 그대로: 끝에서 프레임 추출
            t_frame = max(base_clip.duration - 0.1, 0)
            frame = base_clip.get_frame(t_frame)  # numpy array
            freeze_duration = mp3_duration - base_clip.duration + padding
            freeze = (
                ImageClip(frame)
                .set_duration(freeze_duration)
                .set_fps(base_clip.fps)
                .resize(base_clip.size)  # 원래 크기 유지
            )
        else:
            t_frame = max(base_clip.duration - 0.1, 0)
            frame = base_clip.get_frame(t_frame)
            freeze = (
                ImageClip(frame)
                .set_duration(adding)
                .set_fps(base_clip.fps)
                .resize(base_clip.size)
            )
        extended_video = concatenate_videoclips([base_clip, freeze])
    else:
        extended_video = base_clip

    # 3. 영상에 오디오 입히기
    composite = extended_video.set_audio(delayed_audio)
    
    # 4. 영어 텍스트 자막 추가
    text_clip = create_text_clip(eng_text, composite.duration, composite.size)
    final_clip = CompositeVideoClip([composite, text_clip])
    
    clips.append(final_clip)

# 전체 영상 이어붙이기
final_video = concatenate_videoclips(clips, method="compose")
final_video.write_videofile("result.mp4", fps=24, codec="libx264", preset="medium", ffmpeg_params=["-crf", "18"])