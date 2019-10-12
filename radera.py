def text_to_numeric(*sequence):
    """
    one_use         :  zero    | ten    | eleven | twelve | thirteen | fifteen

    below_ten       :  one     | two    | three  | four   | five     | six      | seven  | eight  | nine

    below_twenty    : below_ten [teen]
                    | one_use

    below_hundred   : twenty  | thirty | forty  | fifty  | sixty    | seventy  | eighty | ninety
                    | [twenty | thirty | forty  | fifty  | sixty    | seventy  | eighty | ninety] below_ten
                    | below_ten

    below_thousand  : below_ten      [hundred  [[and] below_hundred]]
    below_million   : below_thousand [thousand [[and] below_hundred]]
    below_billion   : below_million  [million  [[and] below_thousand]]


    one hundred twenty two                    122
    six thousand five hundred ninty nine    6,599
    eleven thousand                        11,000
    three thousand sixty seven              3,067
    one hundred thousand one              100,001
    two thousand seventeen                  2,017
    six hundred eight                         608
    nineteen thousand fifteen              19,015


    Iterate backwards and add the value to the total. If hundred, thousand, million or billion is encountered,
    multiply it by the following number (since we iterate backwards it'll actually be the preceding number).
    """
    single = {
        'zero'     : 0,
        'one'      : 1,
        'two'      : 2,
        'three'    : 3,
        'four'     : 4,
        'five'     : 5,
        'six'      : 6,
        'seven'    : 7,
        'eight'    : 8,
        'nine'     : 9,
        'ten'      : 10,
        'eleven'   : 11,
        'twelve'   : 12,
        'thirteen' : 13,
        'fourteen' : 14,
        'fifteen'  : 15,
        'sixteen'  : 16,
        'seventeen': 17,
        'eighteen' : 18,
        'nineteen' : 19,
        'twenty'   : 20,
        'thirty'   : 30,
        'forty'    : 40,
        'fifty'    : 50,
        'sixty'    : 60,
        'seventy'  : 70,
        'eighty'   : 80,
        'ninety'   : 90,
    }
    multipliers = {
        'hundred' : 100,
        'thousand': 1000,
        'million' : 1000000,
    }

    total = 0
    index = len(sequence) - 1
    previous = 0

    while index >= 0:

        number = sequence[index]

        if number in single:
            value = single[number]

        elif number in multipliers:
            value = multipliers[number]

            index -= 1
            number = sequence[index]
            while number in multipliers:
                index -= 1
                other_value = multipliers[number]

                if value < other_value:    # Example 'one thousand hundred'
                    raise Exception('value={}, other_value={}'.format(value, other_value))

                value *= other_value
                number = sequence[index]

            value = (single[number] * value)

        else:
            raise Exception('BUHU!')

        if value < previous:
            raise Exception('value={}, previous={}'.format(value, previous))

        previous = value
        total += value
        index -= 1

    return total


print(text_to_numeric(*'one hundred twenty two               '.split()))
print(text_to_numeric(*'six thousand five hundred ninety nine'.split()))
print(text_to_numeric(*'eleven thousand                      '.split()))
print(text_to_numeric(*'three thousand sixty seven           '.split()))
print(text_to_numeric(*'one hundred thousand one             '.split()))
print(text_to_numeric(*'two thousand seventeen               '.split()))
print(text_to_numeric(*'six hundred eight                    '.split()))
print(text_to_numeric(*'nineteen thousand fifteen            '.split()))



# print(text_to_numeric(*'one hundred two twenty               '.split()))
# print(text_to_numeric(*'six thousand hundred ninety nine     '.split()))
# print(text_to_numeric(*'three sixty seven           '.split()))
# print(text_to_numeric(*'six hundred eight two                   '.split()))  # FIX!
# print(text_to_numeric(*'nineteen thousand fifteen            '.split()))
# print(text_to_numeric(*'nine hundred sixty five million two thousand two hundred seventy two'.split()))     # FIX!
# print(text_to_numeric(*'nine hundred sixty five million eleven thousand two hundred seventy two'.split()))  # FIX!