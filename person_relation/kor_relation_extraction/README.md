# Relation Extraction Model for Korean
KAIST Relation 데이터셋을 BERT로 학습시킨 한국어 관계 추출 모델입니다.

## Requirements
- Python 3
- torch >= 1.4.0
- transformers >= 2.7.0

```
  pip install torch
  pip install transformers
```

## 실행
- predict.py 파일을 실행하면 됩니다.
- Arguments
  `--input_file`: 관계 추출할 파일명 (default: "sample_pred_in.txt")
  `--output_file`: 관계 추출 결과를 저장할 파일명 (default: "sample_pred_out.txt")
  `--model_dir`: 관계 추출 모델 위치 (default: "./model")

```
  python predict.py --input_file {YOUR_INPUT_FILE} --output_file {YOUR_OUTPUT_FILE}
```
## Relation Labels
- KAIST 관계 데이터의 관계는 다음과 같습니다.
  - ["producer", "country", "subsequentWork", "operatedBy", "city", "religion", "leaderName", "managerClub",
      "father", "mother", "battle", "publisher", "parent", "related", "author", "leader", "type",
      "servingRailwayLine", "opponent", "regionServed", "division", "combatant", "relative", "isPartOf",
      "era", "recordLabel", "manufacturer", "knownFor", "associatedMusicalArtist", "currentTeam", "order",
      "child", "family", "associatedAct", "product", "channel", "phylum", "director", "genre", "keyPerson",
      "starring", "league", "spouse", "region", "bandMember", "dynasty", "notableWork", "artist", "part"]
- 관계 결과는 integer 형태의 라벨로 출력됩니다.
- 각 라벨은 다음과 같이 매칭됩니다.
  - "0": "producer",
  - "1": "country",
  - "2": "subsequentWork",
  - "3": "operatedBy",
  - "4": "city",
  - "5": "religion",
  - "6": "leaderName",
  - "7": "managerClub",
  - "8": "father",
  - "9": "mother",
  - "10": "battle",
  - "11": "publisher",
  - "12": "parent",
  - "13": "related",
  - "14": "author",
  - "15": "leader",
  - "16": "type",
  - "17": "servingRailwayLine",
  - "18": "opponent",
  - "19": "regionServed",
  - "20": "division",
  - "21": "combatant",
  - "22": "relative",
  - "23": "isPartOf",
  - "24": "era",
  - "25": "recordLabel",
  - "26": "manufacturer",
  - "27": "knownFor",
  - "28": "associatedMusicalArtist",
  - "29": "currentTeam",
  - "30": "order",
  - "31": "child",
  - "32": "family",
  - "33": "associatedAct",
  - "34": "product",
  - "35": "channel",
  - "36": "phylum",
  - "37": "director",
  - "38": "genre",
  - "39": "keyPerson",
  - "40": "starring",
  - "41": "league",
  - "42": "spouse",
  - "43": "region",
  - "44": "bandMember",
  - "45": "dynasty",
  - "46": "notableWork",
  - "47": "artist",
  - "48": "part"