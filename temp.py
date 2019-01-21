orcale_seg = "results/unk-cond-seg_cost.txt"
greedy_seg = "results/unk-cond-greedy-rnnlm.txt"
corpus_batch = "results/corpus-batch.txt"

with open(orcale_seg, 'r') as oin,open(greedy_seg, 'r') as gin,open(corpus_batch, 'r') as cin:
	odata = oin.readlines()
	gdata = gin.readlines()
	cdata = cin.readlines()

# odata = [i.strip() for i in odata]
# cdata = [i.strip() for i in cdata]
# gdata = [i.strip() for i in gdata]

with open("results/combined-seg-oracle-greedy.txt", 'w') as fout:
	for i, j, k in zip(odata, cdata, gdata):
		fout.write(i)
		fout.write(k)
		fout.write(j)
