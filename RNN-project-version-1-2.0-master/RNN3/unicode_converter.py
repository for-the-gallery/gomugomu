import math

## 한글 글자를 유니코드로 대치시키는 코드, return 0 면 한글 글자가 아님.
def chr2unicode(word):
	uni_start = 44032
	uni_end = 55203


	test_unicode = uni_start
	while 1:
		if word == chr(test_unicode):
			return test_unicode
		else:
			test_unicode = test_unicode + 1
			if test_unicode > uni_end:
				return 0

## 한글 글자를 초성 중성 종성으로 구분
def chr_diss(word):
	uni_start = 44032

	first_arr = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ',
		'ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
	second_arr = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ',
		'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
	third_arr = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
		'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ',
		'ㅌ','ㅍ','ㅎ']

	first_size = len(first_arr)
	second_size = len(second_arr)
	third_size = len(third_arr)

	rela_num = chr2unicode(word)-uni_start
	first_index = math.floor(rela_num/second_size/third_size)
	second_index = math.floor((rela_num-first_index*second_size*third_size)/third_size)
	third_index = rela_num - first_index*second_size*third_size - second_index*third_size

	return first_arr[first_index],second_arr[second_index],third_arr[third_index]

def chr_ass(component):
	uni_start = 44032

	first_arr = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ',
		'ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
	second_arr = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ',
		'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
	third_arr = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
		'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ',
		'ㅌ','ㅍ','ㅎ']

	first_size = len(first_arr)
	second_size = len(second_arr)
	third_size = len(third_arr)

	if not(component[0] == ''):
		first_index = first_arr.index(component[0])
		second_index = second_arr.index(component[1])
		third_index = third_arr.index(component[2])

		word_number = uni_start + third_index + second_index*third_size + first_index*second_size*third_size
		return chr(word_number)	
	else:
		return component[2]	

	