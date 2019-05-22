# Test set의 특징
- Test set에도 결측값이 있어서 전처리가 필요하다.
- Train set 과 마찬가지로 Age와 Cabin에 결측값이 있고, Fare에서 결측값이 1개[152번째 idx) 등장했다.

# 앞으로의 실험 방식
- 모델을 먼저 정한다.
- validation set은 k-fold로 한다. 
  - 따로 validation을 분리할 필요없고 굳이, 분포를 비교해가며 확인할 필요없다.
  - 단, 시간이 오래걸린다.
- 가장 간단한 방식으로 전처리를 먼저해보고 각 모델의 default값을 통해서 실험을 한다. (이게 baseline)
- 이후 hyperparameter 조정이 필요한 모델은 hyperparameter에 대한 값을 조정하며 실험한다. 
  - 각각의 parameter값은 기록할것.
- 이후 feature engineering을 통해 추가로 성능을 올린다.

# 실험을 하면서 필요한 정보
- Titanic의 평가방법인 Accuracy
  - 단, Accuracy에 대한 decision boundary는 최적의 AUC값에서 찾을 거기 때문에 이에대한 threshold값도 제시한다.
- AUC
- Prediction에 사용한 Threshold
- 사용 seed
- 그 외 모델 또는 K-fold의 hyperparmeter
 
 
 
# 실험
## Train set
###  전처리
#### 방법1
- 결측값 처리
  - 나이의 결측값 모두 99
  - Embarked의 경우 nan이 str로 되어있어서 None으로 바꾸고 결측값이 있는 행 제외
- 변수 제외 
  - 'Cabin','Name','Ticket','PassengerId'
  - 당장 사용하기 애매해서 제외했음
- 더미변환
  - 'Sex','Embarked'
- 타입변환
  - 'Age' : float -> int
- 마지막 남은 Dataframe의 shape: (889,9)
- 남은 Target : {0: 549, 1: 342}


## Test set
### 전처리
#### 방법1
- Fare에 대한 결측값
  - 분포를 보니 한쪽에 치우쳐진 모양이라서 중앙값으로 대체함
- Age에 대한 결측값
  - 결측값은 모두 99로 대채함
- 변수제외
  - 'Cabin','Name','Ticket','PassengerId'
  - 전처리 방법1에 맞춤
- 더미변환
  - 'Sex','Embarked'
- 타입변환
  - 'Age' : float -> int


## Model 
### Logistic Regression
- submission 1
  - 사용 전처리
    - Train set : 방법1, Test set: 방법1
    - Train_test_split : seed = 223, size = 0.3
  - 사용 threshold
    - 0.3662619979499872
  - Public score 
    - 0.75598

- submission 1
  - 사용 전처리
    - Train set : 방법1, Test set: 방법1
    - Train_test_split : seed = 223, size = 0.3
  - 사용 threshold
    - 0.5
  - Public score 
    - 0.80382




## 일지
- submission 1,2 가 단순히 threshold를 바꾸어줬을 뿐인데 결과에 엄청난 차이를 보인다.
- threshold를 어떻게 해야할지를 더욱 고민해봐야하고, 역시 Test set을 하나 따로 나누는 것 보다는 K-fold를 시도해 보는 것이 더 일반화하기에 적합한 것 같다.

