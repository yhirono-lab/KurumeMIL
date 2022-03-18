# Attention-based Multiple Instance Learning with YOLO

通常のAMILのプログラム．   
MILYOLOと構造はほぼ同じだが，MILYOLOの方がみやすいプログラムだと思います．    
パッチ画像は保存せず，/Raw/Kurume_Dataset/svs に保存されたsvsファイルから逐次パッチ画像を作成する．

## 実行環境
ライブラリバージョン
- python 3.8.10
- numpy 1.21.2
- opencv-python 4.5.3.56
- openslide-python 1.1.2
- pillow 8.3.1
- torch 1.9.0
- torchvision 0.10.0

使用したマシン
- マシン noah
- CUDA Version 11.4
- Driver Version 470.86

## OpenSlide のインストール
OpenSlideというライブラリをマシンにインストールしないとパッチ画像の作成ができない．  
(通常であれば，入っている環境に設定してくれていると思う)  
python用のライブラリをダウンロード  
```
apt-get install python-openslide
apt-get install openslide-tools
pip3 install openslide-python
```

## ファイル構造
必要なデータセット
```
画像データ
root/
　　├ Raw/Kurume_Dataset/svs
　　└ Dataset/Kurume_Dataset/hirono

決定木データ
../KurumeTree
    ├ kurume_tree
    |   ├ 1st
    |   |   ├ Simple   
    |   |   └ Full
    |   |       ├ tree/                     #決定木の図の保存
    |   |       ├ ...  
    |   |       └ unu_depthX/leafs_data     #各深さにおける各葉に分類されたデータセット
    |   |                                   #MILのデータ読み込みに使う
    |   ├ 2nd        
    |   └ 3rd        
    └ normal_tree
        └ 同上
```

必要なソースコード

```
Source/
　　├ dataloader_svs.py     #データローダーの作成のプログラム
　　├ dataset_kurume.py     #データセットの作成のプログラム
　　├ draw_heatmap.py       #Attentionの可視化のプログラム
　　├ make_log_Graphs.py    #訓練時のグラフを描画するプログラム
　　├ make_yolo_dataset.py  #YOLOの学習に必要なパッチ画像を抽出するプログラム
　　├ MIL_test.py           #テスト用プログラム
　　├ MIL_train.py          #訓練用プログラム
　　├ model.py              #AMILで使用するモデルのプログラム
　　├ run.py                #必要なプログラムを一括実行するプログラム
　　└ utils.py              #汎用的な関数のプログラム
```

プログラムの実行順序と依存関係(run_train.pyで一括実行)
```
Source/
　　├ 1.train.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ model.py
　　|　 
　　├ 2.test.py
　　| 　├ utils.py
　　| 　├ dataset_kurume.py
　　|　 ├ dataloader_svs.py
　　|　 └ model.py
　　|　 
　　├ 3.make_log_Graphs.py
　　| 　└ utils.py
　　|　 
　　└ 4.draw_heatmap.py
　　  　└ utils.py
```

各プログラムには実行時にパラメータの入力が必要
run_train.pyで一括実行する際は，train.py,test.pyの*の付いたパラメータに注意
```
train               データのクロスバリデーションの訓練番号を指定 例：123
valid               データのクロスバリデーションの検証番号を指定 例：4
--depth             決定木の深さを指定 例：1
--leaf              決定木の指定した深さの葉の指定 例：01
--data              データ数の選択 (1st, 2nd, 3rd)
--mag               拡大率の選択(40x以外はデータセットがないから不可) 例：40x
--model             MILの特徴抽出器の選択 (vgg16 or vgg11)
--name              データセット名の指定(normal_tree,subtypeの時に有効)
                        * Simple : 例 DLBCL, FL
                        * Full : 例 DLBCL-GCB, FL-grade1
*--num_gpu          使用するGPU番号の数の指定
-c, --classify_mode 識別する決定木の種類の選択 
                        * normal_tree : エントロピーによる決定木で識別
                        * kurume_tree : 免疫染色による決定木で識別
                        * subtype : サブタイプ名別で識別(5種類)
-l, --loss_mode     損失関数の選択
                        * CE : Cross Entropy Loss
                        * ICE : Inversed Cross Entropy Loss (重み付けCE)
                        * focal : Focal Loss
                        * focal-weight : Weighted Focal Loss (重み付けFocal Loss)
                        * LDAM : LDAM Loss
--lr                学習率の指定 例：0.00005
-C, --constant      LDAM損失使用時のパラメータ指定(0～0.5くらい) 例：0.2
-g, --gamma         focal損失使用時のパラメータ指定 例：1.0
-a, --augmentation  (flag)回転・反転によるaugmentationの有無
*-r, --restart      (flag)再学習するか否か(上手くできているか不明)
--fc                (flag)ラベル予測部分のみを学習するか否か
--reduce            (flag)データ数が多い時に多すぎるラベルを減らすか否か
```
*注意1    
--restartはMILYOLOと違いちゃんと動作するはずです  
*注意2  
画像svsファイル名の変更や，パラメータの名前の変更によってエラーが発生する可能性あるかもしれないです  
*注意3  
MILYOLOと違い検証用のバリデーションの番号を指定する必要があります    

各プログラムを実行するとパラメータごとに結果が保存される．
```
\train_log          #訓練時の損失などの値が記録される(MIL_train.py)
\test_result        #テストデータによるバッグごとの識別結果が記録される(MIL_test.py)
\model_params       #モデルのパラメータが保存される(MIL_train.py)
\graphs             #訓練時の損失などのグラフの図が保存される(make_log_Graphs.py)
\attention_patch    #テストデータの識別結果からアテンションの高いパッチが保存される(draw_heatmap.py)
\attention_map      #テストデータの識別結果からアテンションのカラーマップが保存される(draw_heatmap.py)
```

#### ファイル名の変更の対応関係
ファイル名の修飾子を分かりやすいものに変えました  
その影響で過去の実験結果を使ってMIL_test.pyやmake_log_Graphs.pyやdraw_heatmap.pyをそのまま実行できない  
実行するときは対象のファイル名を以下を参考にして変更してください  
(修飾無し)は，修飾子をつけたあとに_(アンダーバー)をつけてください

データセットのバージョン
```
(修飾無し) → 1st  
add_ → 2nd  
New : 3rd  (実験したことのないのでエラーに注意)
```

決定木の名前
```
leaf → normal_tree
new_tree → kurume_tree
subtype → subtype
```

モデルの名前
```
(修飾無し) → vgg16
vgg11 → vgg11
```

損失関数の名前
```
normal → CE
invarse → (廃止)
myinvarse → ICE
LDAM → LDAM
focal → focal
focal-weight → focal-weight
```

