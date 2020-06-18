온도 추정 모델
-------------------------------

개요
+ 주어진 기상청 공공데이터를 이용해 생성한 RNN 모델을 기반으로 추후 80일 간의 온도 예측
+ DACON에서 주최한 경진대회 참여   
Link : [[DACON]공공 데이터 활용 온도 추정 AI 경진대회](https://dacon.io/competitions/official/235584/overview/)

프로젝트 환경
+ IDE : Pycharm, Rstudio, Anaconda, Jupyter Notebook
+ OS : Windows 10
+ Language : Python 3.7.4, R 3.6.3
+ Framework : TensorFlow 2.1 GPU, Keras
+ 팀원 : 5명

진행 과정
+ 2020.03.13 ~ 2020.03.17 : 데이터 특징과 상관관계 분석
+ 2020.03.18 ~ 2020.03.20 : 모델 구성
+ 2020.03.21 ~ 2020.03.30 : 모델 fine tuning, 데이터 전처리
+ 2020.03.31 ~ : optimizer와 데이터 전처리 가공 방법 도모

개선 사항
+ 더 효과적인 전처리 방법 모색
  + 풍향, 습도
+ 다른 앙상블 모델 적용 시도
  + 모델 개수 조정, hyper parameter 조정
+ 다른 알고리즘, optimizer 모색
  + Layer Normalization, Batch Normalization
  + Nadam, Adaboost, AMS grad 등
