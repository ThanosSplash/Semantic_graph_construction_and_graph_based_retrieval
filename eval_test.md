## Test_1

### Τα αποτελεσματα μου
```text
Prediction = ["A", "B", "C", "D"]
Ground truth = ["C", "X"]

recall@4 = 1/2 = 0.5  , recall@2 = 0/2 
RR = 1/2 
ndcg@4 = dcg@4/idcg@4 = 0.5/1 = 0.5
[0,0,0,1]
dcg@4 = 0*1/log2(2) + 0*1/log2(3) + 1*1/log2(4)= 1/2 = 0.5
[1,0,0,0]
idcg@4 = 1*1/log2(2) = 1
```
### Αποτελεσματα προγραμματος
```text

recall@4 = 0.5, recall@2 = 0/2 = 0
RR = 1/2 
ndcg@4 = dcg@4/idcg@4 = 0.5/1 = 0.5
```
## Test_2
```text
Prediction = ["C", "B", "X", "D"]
Ground truth = ["C", "B", "X"]
```
### Τα αποτελεσματα μου
```text
recall@4 = 3/3 = 1
RR = 1/1 
ndcg@4 = dcg@4/idcg@4 = (1*1/log2(2) + 1*1/log2(3) + 1*1/log2(4))/(1*1/log2(2) + 1*1/log2(3) + 1*1/log2(4)) = 1
[1, 1, 1, 0]
dcg@4 = 1*1/log2(2) + 1*1/log2(3) + 1*1/log2(4) 
[1, 1, 1, 0]
idcg@4 = 1*1/log2(2) + 1*1/log2(3) + 1*1/log2(4) 
MAP@4 = 3/3 = 1 

```
### Αποτελεσματα προγραμματος
```text

recall@4 = 1
RR = 1
ndcg@4 = dcg@4/idcg@4 = 1
MAP@4 = 1
```

## Test_3
```text
Prediction = ["A", "E", "F"]
Ground truth = ["A", "B", "X"]
```
### Τα αποτελεσματα μου
```text
recall@3 = 1/3 = 0.3
RR = 1/1 
ndcg@3 = dcg@3/idcg@3 = 1*log2(2)/1*log2(2) = 1
[1, 0, 0]
dcg@3 = 1*log2(2)
[1, 0, 0]
idcg@3 = 1*log2(2) 
MAP@3 = 0.33333
```
### Αποτελεσματα προγραμματος
```text

recall@3 = 0.33333
RR = 1/1 = 1
ndcg@3 = dcg@4/idcg@3 = 1
MAP@3 = 0.33333
```

## Test_4
```text
Prediction = ["E", "F", "A"]
Ground truth = ["A", "B", "X"]
```
### Τα αποτελεσματα μου
```text
recall@3 = 1/3 = 0.3333
RR = 1/3 = 0.333333 
ndcg@3 = dcg@3/idcg@3 = 0.5/1 = 0.5
[0, 0, 1]
dcg@3 = 1*log2(4) = 0.5
[1, 0, 0]
idcg@3 = 1*log2(2) = 1
MAP@3 = 0.1111111111111111
```
### Αποτελεσματα προγραμματος
```text

recall@3 = 0.3333333333333333
RR = 1/3 = 0.3333333333333333
ndcg@3 = dcg@4/idcg@3 = 0.5
MAP@3 = 0.1111111111111111
```


## Test_4
```text
Prediction = ["A", "I", "E", "D"]
Ground truth = ["C", "B", "X"]
```
### Τα αποτελεσματα μου
```text
recall@4 = 0/3 = 0.0
RR = 0  
ndcg@4 = dcg@4/idcg@4 = 0/0 = 0.0
[0, 0, 0, 0]
dcg@4 = 0.0 
[0, 0, 0, 0]
idcg@4 = 0.0
MAPA@4 = 0.0
```
### Αποτελεσματα προγραμματος
```text

recall@4 = 0.0
RR = 1/3 = 0
ndcg@4 = dcg@4/idcg@3 = 0.0
MAP@4 = 0.0
```