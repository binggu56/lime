def test(n,m, L):
	if m==0:
		return
	L2 = []
	for i in range(n):
		L2 += [i]
		test(n, m-1, L2)
	L += [L2]
	return
List = []
test(3, 3, List)

print List