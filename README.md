# 私の研究に興味を持ってくれた方へ
- 修論を終えてから気づいたのですが，私の研究は「face swapping」という技術に関連しており，先行研究が多くある可能性があります．研究を引き継ぐことを考えている人は，注意してください．
- 私のGitLabのページや修士論文を合わせて見てもらうとコードの理解度が上がるかもしれません．
- https://github.com/d-omote/workspace-main
- https://tduvcl.sharepoint.com/:f:/s/GraduationThesis/Ep-WmPtJobBAiGdZTPNZP_ABuyxlrYX1AS_ZyhXGw0P3MQ?e=C0hm34


# 実行環境
- [x] python(3.10.4)
- [x] visual Studio 2017
- [x] CUDA(11.3)
- [x] CuDNNのインストールも必要かも?(ユーザ登録必要)
- [x] Pytorch(1.11.0+cu113)「pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113」
- [x] CMake(3.25.0-rc2)https://github.com/Kitware/CMake/releases/download/v3.25.0-rc2/cmake-3.25.0-rc2-windows-x86_64.msi
- [x] dlib(19.24.0)「pip install dlib」
- [x] OpenCV(4.6.0)「pip install opencv-contrib-python」
- [x] Numpy(1.22.3)
- [x] imutils(0.5.4)
- [x] PIL(9.1.0)

# コードに関する書きおき
## 無関係なpyファイル
疑似的に絵画から筆跡を検出しようとして，画像を領域分割した名残がありますが，人物合成には使用していません．
- Region.py
- regionColorList.py
- RegionGrow.py

__init__.pyは，GUIを実装しようとして挫折した跡なので，未完成のコードです． 

## 人物合成に使用するpyファイル
-test.py
|-Composit2paintingMask.py
 |-FaceCropping.py
 |-styleTransfer2.py

### test.py
実行用のファイル.プロンプトでこのファイルを実行してください

### Composit2paintingMask.py
人物の顔パーツを絵画に合成するクラス

### FaceCropping.py
人物の顔パーツを写真から切り抜くクラス

### styleTransfer2.py
スタイル変換を行うクラス
(styleTransfer.pyは，Pytorchでスタイル変換しようとして上手くいかなかったものです)

### demo.py
修論発表会でデモを行うために作ったファイルです．
カメラを外付けすれば動くはずです．