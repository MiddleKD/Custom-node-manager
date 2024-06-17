import os
import shutil
import fnmatch

def is_ignored(path, ignore_patterns):
    for pattern in ignore_patterns:
        if pattern.startswith('/'):
            if fnmatch.fnmatch(path, pattern[1:]):
                return True
        elif pattern.endswith('/'):
            if any(fnmatch.fnmatch(part, pattern.rstrip('/')) for part in path.split(os.path.sep)):
                return True
        elif '/' not in pattern:
            if any(fnmatch.fnmatch(part, pattern) for part in path.split(os.path.sep)):
                return True
        else:
            if fnmatch.fnmatch(path, pattern):
                return True
    return False

def copy_directory(src_dir, copyignore_path):
    # .copyignore 파일 읽기
    ignore_patterns = []
    if os.path.exists(copyignore_path):
        with open(copyignore_path, 'r') as file:
            ignore_patterns = file.read().splitlines()

    # 디렉토리 복사
    for root, dirs, files in os.walk(src_dir):
        # 상대 경로 계산
        rel_path = os.path.relpath(root, src_dir)

        # .copyignore에 명시된 경로 제외
        if is_ignored(rel_path, ignore_patterns):
            dirs[:] = []  # 하위 폴더 탐색 중단
            continue

        dest_dir = os.path.join(os.getcwd(), os.path.basename(src_dir), rel_path)

        # 대상 디렉토리 생성
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # 파일 복사
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)

            # .copyignore에 명시된 파일 제외
            if is_ignored(os.path.join(rel_path, file), ignore_patterns):
                continue

            shutil.copy2(src_file, dest_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    args = parser.parse_args()

    source_directory = args.src
    copyignore_file = './.copyignore'
    copy_directory(source_directory, copyignore_file)