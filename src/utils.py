def deco_print(line):
	print('>==================> ' + line)

def deco_print_dict(dic):
	for key, value in dic.items():
		deco_print('{} \t {}'.format(key, value))