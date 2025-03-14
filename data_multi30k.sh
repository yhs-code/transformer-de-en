python data_multi30k.py --pair_dir $1 --dest_dir $2 --src_lang $3 --trg_lang $4

# 新建train_l文件，合并两个文件到一个文件
touch $1/train_l
cat $2/train_src.cut.txt >> $1/train_l
cat $2/train_trg.cut.txt >> $1/train_l

# 生成词表，subword方式,统一用20000个subword，-i 表示输入文件，-s 表示subword数量，-o 表示输出文件，--write-vocabulary 表示输出词表
subword-nmt learn-joint-bpe-and-vocab \
    -i $1/train_l \
    -s 20000 \
    -o $1/bpe.20000 \
    --write-vocabulary $1/vocab

# 应用分词，其中 -c 选项指定bpe模型，< $2/${mode}_src.cut.txt 表示输入文件，> $1/${mode}_src.bpe 表示输出文件
for mode in train val test; do
    subword-nmt apply-bpe -c $1/bpe.20000 < $2/${mode}_src.cut.txt > $1/${mode}_src.bpe
    subword-nmt apply-bpe -c $1/bpe.20000 < $2/${mode}_trg.cut.txt > $1/${mode}_trg.bpe
    echo "Finished applying bpe to ${mode} files."
done
