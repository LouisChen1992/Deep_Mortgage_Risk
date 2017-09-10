def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)

def deco_print_dict(dic):
	for key, value in dic.items():
		deco_print('{} : {}'.format(key, value))
