import pickle
import csv

# 파일명 설정
pkl_filename = "final_result.pkl"
csv_filename = "final_result.csv"

# Pickle 파일 로드
with open(pkl_filename, "rb") as file:
    data = pickle.load(file)

# CSV 파일 저장 로직
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # 데이터 타입별 처리
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):  # 리스트 안에 딕셔너리가 있는 경우 (테이블 형태)
            headers = list(data[0].keys())  # 첫 번째 딕셔너리의 키를 헤더로 사용
            writer.writerow(headers)  # CSV 헤더 작성
            for row in data:
                writer.writerow([row[key] for key in headers])  # 각 행의 값 작성
        else:
            writer.writerow(["Value"])  # 단순 리스트일 경우
            for item in data:
                writer.writerow([item])
    elif isinstance(data, dict):
        writer.writerow(["Key", "Value"])  # 딕셔너리를 Key-Value 형식으로 저장
        for key, value in data.items():
            writer.writerow([key, value])
    else:
        writer.writerow(["Data"])  # 기타 데이터 타입 처리
        writer.writerow([data])

print(f"✅ Pickle data successfully saved to {csv_filename}")
