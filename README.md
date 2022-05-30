# simContext
This project is matching algorithm base on combined six features(event, person, time, location, emotion, relation)  from the Korean novels.   
We seperated two NLP tasks and extracted hidden feature (identical size 768) from sentences using koBERT.
- Named Entity Recognition (NER) : time, location, person, event
- Sequence Classification : emotion, relation

Each book has one combined context feature by averaging sentence features.  
This combined context feature is used to compute cosine similarity for matching novels.

## Environment
- cuda 10.1
- python 3.7.9
- torch 1.7.1
- torchvision 0.8.2
```
pip install -r requirements.txt
```


## Prepare book dataset
Book dataset are .json format and save them path to 'book_dataset/book-json'   
Example(.json ) : 알퐁스도데_마지막수업
```
{"texts":[
    {"page":"1","text":"그날 나는 등교가 무척 늦어졌어요. 게다가 아멜 선생님이 분사법(分詞法)에 관해 질문하겠다고 하셨는데, 나는 전혀 모르고 있었기에 책망을 들을까 봐 꽤 겁이 났어요. 그래서 나는 차라리 학교를 가지 않고 벌판이나 싸다닐까 하고 생각해 보기도 했어요.  날씨는 활짝 개여 있었어요. 숲가에서는 티티새가 떼지어 지저귀고, 제재소 뒤 리뻬르 들에서는 프러시아 군대들이 훈련하는 소리가 들려왔어요. 이것들이 나에게는 분사법보다 더 마음에 들었지만, 나는 참고 학교로 뛰어갔어요.  면사무소 앞을 지나는데, 사람들이 철책을 두른 게시판 앞에 모여 있는 것이 내 눈에 띄었어요. 이태 전부터 패전이니 징발이니 하는 프러시아 군사령부의 여러 가지 언짢은 뉴스들은, 다 이곳에서 나왔던 거예요.  \"또 무슨 일이 일어나려는 걸까?\"  나는 여전히 뛰어가면서 생각했어요. 내가 광장을 지날 때였어요. 대장간집 와슈테르 영감이 조수와 함께 게시판을 들여다보다가 말하는 것이었어요.  \"얘야, 그렇게 서두를 거 없다. 지각은 하지 않을 테니까.\"  나는 영감이 나를 놀리는 줄로만 알고 숨을 몰아쉬면서 아멜 선생네의 비좁은 마당에 뛰어들어갔어요.  평소에는 수업이 시작되면 으례, 책상 뚜껑을 여닫는 소리며 책을 잘 외우기 위해 귀를 막고 커다란 소리로 읽어 내려가는 소리, 그리고 '좀 조용히 해' 하고 책상을 마구 두들겨 대는 선생님의 회초리 소리 등이 떠들썩하게 한길까지 들려와, 나는 그 법석을 부리는 통에 살짝 내 자리로 가려고 하다가 주춤하곤 했지요. 그런데 그날은 일요일 아"},
    {"page":"2","text":"침처럼 조용했어요. 열린 창문을 통해, 진작 제 자리에 앉은 아이들과 그 무서운 회초리를 팔에 끼고 서성대는 아멜 선생님이 보였어요. 나는 이처럼 조용한 가운데 문을 열고 들어가 앉는 도리밖에 없었어요. 내 얼굴이 붉어지고 가슴이 두근거렸을 거라고 생각하세요? 천만에요, 선생님은 아무 화도 내지 않고 날 바라보더니 부드러운 목소리로 말했어요.  \"프란츠, 어서 네 자리로 가서 앉거라. 우리는 그냥 수업을 시작할 뻔했구나.\"  나는 얼른 의자 너머의 내 자리로 가 앉았어요. 두려운 마음이 좀 사라지자, 선생님이 푸른 프록코트 차림에, 가슴에는 주름 잡힌 장식을 달고, 수놓은 검은 비단으로 된 둥근 모자를 쓰고 있는 것이 보였어요. 그것은, 장학관의 시찰이 있거나 시상식 같은 것이 있는 때에만 입는 예복 차림이었어요. 그리고 교실 전체에 여느 때와는 다른 엄숙한 분위기가 감돌고 있었어요. 가장 놀란 것은, 평소에 비어 있던 교실 안쪽 의자에 마을 사람들이 조용히 앉아 있는 것이었어요. 머리에 삼각모를 쓴 오제 영감과 전면장, 우체부, 그밖에 많은 사람들이 앉아 있었어요. 그들은 저마다 슬픈 표정들이었어요. 오제 영감은 모서리가 다 해어진 프랑스어 초보 교재를 무릎 위에 펴놓고, 그 위에 커다란 안경을 올려 놓고 있었어요.  나는 이런 광경을 보고 그저 어리둥절한 얼굴을 하고 있었어요. 아멜 선생님이 교단에 올라가더니, 나를 맞아줄 때와 다름없는 부드럽고 엄숙한 어조로 말했어요.  \"여러분, 이것이 내 마지막 수업이에요. 베를린에서 알사스와 로렌의 학교에서는 독일어만 가르치라는 지시가 내렸어요. 내일 새 선생님이 오십니다. 오늘로서 프랑스어 공부는 끝입니다. 명심해 들어요.\" "},
    ...  
    {"page":"6","text":"어 읽고 있었어요. 그도 무척 열심이었어요. 그의 목소리는 감동한 나머지 떨리고 있었어요. 그의 책 읽는 소리가 하도 우스워 우리는 웃어야 할지 울어야 할지 알 수 없었어요. 아, 나는 이 마지막 수업을 평생 잊을 수가 없겠지요. 성당의 괘종시계가 열두 시를 치더니 이어 앙젤뤼의 종소리가 들려 왔어요. 때마침 교실 창문 아래로 훈련을 끝내고 돌아오는 프러시아군들의 나팔소리가 들려 왔어요. 아멜 선생님은 매우 창백한 얼굴을 하고 교단에서 일어났어요. 선생님이 그렇게 커 보일 수가 없었어요.  \"여러분, 나는...... 나는!......\"  하고 선생님은 말씀했어요.  선생님의 목줄을 무엇인가가 죄이고 있었던 거예요. 선생님은 말을 다 끝맺지 못했어요.  선생님은 흑판을 향해 돌아서더니, 백묵을 쥐고 커다란 글씨로 이렇게 쓰는 것이었어요.  '프랑스, 만세!'  선생님은 벽에 이마를 댄 채 한참 계시더니, 우리에게 손짓하면서 알려 주는 것이었어요.  \"끝났다...... 다들 돌아가거라!\" "}],
 "title":"마지막수업"}
```


## Pre-trained model
Download pre-trained KoBERT model from the [official github](https://github.com/SKTBrain/KoBERT).
- NER
- Emotion
- Relation


## Combine features
Extract six features from the books using each module. Then, combine features and save them to numpy format for similarity.
```
python merge_6feature.py --book_dir 'book_dataset/book-json' --feature_dir 'book_dataset/book-feature'
```


## Search
Search TOP10 results based on cosine similarity between a query and all book features.
```
python search_top10.py --book_feature_dir 'book_dataset/book-feature'
                       --book_meta 'book_dataset/number_title_id.txt' 
                       --query '알퐁스도데_마지막수업'
```

![result image](result.png "search result")