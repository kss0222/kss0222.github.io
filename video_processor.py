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

clips = []
padding = 0.5
audio_delay = 0.5

# 원본 해상도를 유지하면서 비디오 처리
for i, (video_path, audio_path) in enumerate(zip(card_videos, english_captions)):
    # 비디오 클립 로드
    video_clip = VideoFileClip(video_path)
    
    # 오디오 클립 로드
    audio_clip = AudioFileClip(audio_path)
    
    # 원본 비디오의 정보 가져오기
    original_fps = video_clip.fps
    original_size = video_clip.size
    original_duration = video_clip.duration
    
    # get_frame을 사용해서 원본 해상도 유지
    def make_frame(t):
        # 원본 비디오에서 프레임 추출
        if t < original_duration:
            return video_clip.get_frame(t)
        else:
            # 비디오가 끝나면 마지막 프레임 유지
            return video_clip.get_frame(original_duration - 0.01)
    
    # 오디오 길이에 맞춰 비디오 길이 조정
    total_duration = audio_clip.duration + audio_delay + padding
    
    # ImageClip을 사용해서 원본 크기와 fps 명시
    processed_video = VideoClip(make_frame, duration=total_duration)
    processed_video = processed_video.set_fps(original_fps)
    
    # 오디오 지연 적용
    delayed_audio = audio_clip.set_start(audio_delay)
    
    # 비디오와 오디오 합성
    final_clip = processed_video.set_audio(delayed_audio)
    
    # 패딩 추가 (앞뒤로 빈 시간)
    if padding > 0:
        # 앞에 패딩 추가
        padding_clip = VideoClip(lambda t: video_clip.get_frame(0), duration=padding)
        padding_clip = padding_clip.set_fps(original_fps)
        
        # 뒤에 패딩 추가
        end_padding_clip = VideoClip(lambda t: video_clip.get_frame(original_duration - 0.01), duration=padding)
        end_padding_clip = end_padding_clip.set_fps(original_fps)
        
        # 전체 클립을 패딩과 함께 연결
        final_clip = concatenate_videoclips([padding_clip, final_clip, end_padding_clip])
    
    clips.append(final_clip)
    
    # 메모리 정리
    video_clip.close()
    audio_clip.close()
    
    print(f"Processed clip {i+1}/{len(card_videos)}: {os.path.basename(video_path)}")

# 모든 클립을 하나로 연결
print("Concatenating all clips...")
final_video = concatenate_videoclips(clips)

# 최종 비디오 저장
output_path = "/content/drive/MyDrive/final_output.mp4"
print(f"Saving final video to: {output_path}")

final_video.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',
    temp_audiofile='temp-audio.m4a',
    remove_temp=True,
    fps=original_fps  # 원본 fps 유지
)

# 메모리 정리
final_video.close()
for clip in clips:
    clip.close()

print("Video processing completed!")