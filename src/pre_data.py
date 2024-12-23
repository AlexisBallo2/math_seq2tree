# coding: utf-8

import random
import json
import copy
import re
from src.expressions_transfer import *
import sympy as sp
from sympy.solvers import solve
import math
import inflect



PAD_token = 0 

p = inflect.engine()
replace = {}

replace['zero'] = 0
replace['thirty nine'] = 39
replace['sixteen'] = 39
replace['eighteen'] = 18
replace["4teen"] = 14
replace['twice'] = 2
replace['quarter'] = 0.25
replace['thrice'] = 3
replace['half'] = 0.5
replace['first'] = 1
replace['triple'] = 3
replace['dozen'] = 12
replace['double'] = 2
replace['thirds'] = 0.33
replace['4ths'] = 0.25
replace['4th'] = 0.25
# replace['1-4th'] = 0.25
replace['fourths'] = 0.25
replace['fifth'] = 0.2
replace['third'] = 0.33
replace['fourteen'] = 14
replace['306,000'] = 306000
replace['8,200'] = 8200 
for i in range(1, 101):
    word = p.number_to_words(i)
    replace[word] = i



class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            # if word == "^":
            #     print()
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def remove_token_from_vocab(self, token):
        if token in self.index2word:
            index = self.index2word.index(token)
            del self.index2word[index]
            del self.word2count[token]
            del self.word2index[token]
            self.n_words -= 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums, vars, useCustom):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        if useCustom:
            self.index2word = self.index2word + generate_num + vars + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        else:
            self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["UNK"]

        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
    def ids_to_tokens(self, ids):
        output = [] 
        for i in ids:
            output.append(self.index2word[i])
        return output

def load_DRAW_data(filename, filter = None):  # load the json data to list(dict()) for MATH 23K
    print("Reading file...")
    f = open(filename, encoding="utf-8")
    data = json.loads(f.read())
    if filter:
        data = [d for d in data if d['dataset'] == filter]
    # finalData = []
    # for ele in data:
    #     if len(ele['lEquations']) == 1:
    #         finalData.append(ele)
    
    return data

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    # for each line
    for i, s in enumerate(f):
        # build string of 7 lines
        js += s
        i += 1

        if i % 7 == 0:  # every 7 line is a json
            # convert 7 lines to json 
            data_d = json.loads(js)
            # if km/h is in the equation, remove it
            if "千米/小时" in data_d["equation"]:
                # ex: '{
                # "id":"10431",
                # "original_text":"The speed of a car is 80 kilometers per hour. 
                    #   It can be written as: how much. Speed ​​* how much = distance.",
                # "segmented_text":" The speed of a car is 80 kilometers per hour, which can be written as: how much. speed * how much = distance. ",
                # "equation":"x=80 kilometers per hour",
                # "ans":"80"
                # }'
                # -> remove " kilometers per hour"
                # = x=80
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


# # remove the superfluous brackets
# def remove_brackets(x):
#     y = x
#     if x[0] == "(" and x[-1] == ")":
#         x = x[1:-1]
#         flag = True
#         count = 0
#         for s in x:
#             if s == ")":
#                 count -= 1
#                 if count < 0:
#                     flag = False
#                     break
#             elif s == "(":
#                 count += 1
#         if flag:
#             return x
#     return y


# def load_mawps_data(filename):  # load the json data to list(dict()) for MAWPS
#     print("Reading lines...")
#     f = open(filename, encoding="utf-8")
#     data = json.load(f)
#     out_data = []
#     for d in data:
#         if "lEquations" not in d or len(d["lEquations"]) != 1:
#             continue
#         x = d["lEquations"][0].replace(" ", "")

#         if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
#             v = d["lQueryVars"][0]
#             if v + "=" == x[:len(v)+1]:
#                 xt = x[len(v)+1:]
#                 if len(set(xt) - set("0123456789.+-*/()")) == 0:
#                     temp = d.copy()
#                     temp["lEquations"] = xt
#                     out_data.append(temp)
#                     continue

#             if "=" + v == x[-len(v)-1:]:
#                 xt = x[:-len(v)-1]
#                 if len(set(xt) - set("0123456789.+-*/()")) == 0:
#                     temp = d.copy()
#                     temp["lEquations"] = xt
#                     out_data.append(temp)
#                     continue

#         if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
#             continue

#         if x[:2] == "x=" or x[:2] == "X=":
#             if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
#                 temp = d.copy()
#                 temp["lEquations"] = x[2:]
#                 out_data.append(temp)
#                 continue
#         if x[-2:] == "=x" or x[-2:] == "=X":
#             if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
#                 temp = d.copy()
#                 temp["lEquations"] = x[:-2]
#                 out_data.append(temp)
#                 continue
#     return out_data


# def load_roth_data(filename):  # load the json data to dict(dict()) for roth data
#     print("Reading lines...")
#     f = open(filename, encoding="utf-8")
#     data = json.load(f)
#     out_data = {}
#     for d in data:
#         if "lEquations" not in d or len(d["lEquations"]) != 1:
#             continue
#         x = d["lEquations"][0].replace(" ", "")

#         if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
#             v = d["lQueryVars"][0]
#             if v + "=" == x[:len(v)+1]:
#                 xt = x[len(v)+1:]
#                 if len(set(xt) - set("0123456789.+-*/()")) == 0:
#                     temp = d.copy()
#                     temp["lEquations"] = remove_brackets(xt)
#                     y = temp["sQuestion"]
#                     seg = y.strip().split(" ")
#                     temp_y = ""
#                     for s in seg:
#                         if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
#                             temp_y += s[:-1] + " " + s[-1:] + " "
#                         else:
#                             temp_y += s + " "
#                     temp["sQuestion"] = temp_y[:-1]
#                     out_data[temp["iIndex"]] = temp
#                     continue

#             if "=" + v == x[-len(v)-1:]:
#                 xt = x[:-len(v)-1]
#                 if len(set(xt) - set("0123456789.+-*/()")) == 0:
#                     temp = d.copy()
#                     temp["lEquations"] = remove_brackets(xt)
#                     y = temp["sQuestion"]
#                     seg = y.strip().split(" ")
#                     temp_y = ""
#                     for s in seg:
#                         if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
#                             temp_y += s[:-1] + " " + s[-1:] + " "
#                         else:
#                             temp_y += s + " "
#                     temp["sQuestion"] = temp_y[:-1]
#                     out_data[temp["iIndex"]] = temp
#                     continue

#         if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
#             continue

#         if x[:2] == "x=" or x[:2] == "X=":
#             if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
#                 temp = d.copy()
#                 temp["lEquations"] = remove_brackets(x[2:])
#                 y = temp["sQuestion"]
#                 seg = y.strip().split(" ")
#                 temp_y = ""
#                 for s in seg:
#                     if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
#                         temp_y += s[:-1] + " " + s[-1:] + " "
#                     else:
#                         temp_y += s + " "
#                 temp["sQuestion"] = temp_y[:-1]
#                 out_data[temp["iIndex"]] = temp
#                 continue
#         if x[-2:] == "=x" or x[-2:] == "=X":
#             if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
#                 temp = d.copy()
#                 temp["lEquations"] = remove_brackets(x[2:])
#                 y = temp["sQuestion"]
#                 seg = y.strip().split(" ")
#                 temp_y = ""
#                 for s in seg:
#                     if len(s) > 1 and (s[-1] == "," or s[-1] == "." or s[-1] == "?"):
#                         temp_y += s[:-1] + " " + s[-1:] + " "
#                     else:
#                         temp_y += s + " "
#                 temp["sQuestion"] = temp_y[:-1]
#                 out_data[temp["iIndex"]] = temp
#                 continue
#     return out_data

# for testing equation
# def out_equation(test, num_list):
#     test_str = ""
#     for c in test:
#         if c[0] == "N":
#             x = num_list[int(c[1:])]
#             if x[-1] == "%":
#                 test_str += "(" + x[:-1] + "/100.0" + ")"
#             else:
#                 test_str += x
#         elif c == "^":
#             test_str += "**"
#         elif c == "[":
#             test_str += "("
#         elif c == "]":
#             test_str += ")"
#         else:
#             test_str += c
#     return test_str

variableHierarchy = ['X', 'Y', 'Z', 'A', 'B']
def transfer_num(data, setName, useCustom, useEqunSolutions):  # transfer num into "NUM"
    print("Transfer numbers...")
    # number regex
    # pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pattern = re.compile("-?\d*\(\d+/\d+\)\d*|-?\d+\.\d+%?|-?\d+%?")
    var_pattern = re.compile("([xyz])(?:[\+\-\*/])([xyz])(?:\s*=\s*\d+)?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    vars = []
    copy_nums = 0
    for d in data:
        pairNumMapping = {}
        # current_equation_vars = ['X']
        # numbers in this problem's text
        nums = []
        # text after masking
        input_seq = []
        # break up segmented text into each word
        if setName == "MATH":
            seg = d["segmented_text"].strip().split(" ")
        elif setName == "PEN":
            # seg = d["oldText"].strip().split(" ")
            seg = d["text"].strip().split(" ")
        else: 

            seg = d["sQuestion"].strip()
            # replace = { "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven" : 11 }
            seg = seg.lower()
            for k,v in replace.items():
                seg = seg.replace(k, str(v))
            seg = seg.split(" ")

        # strip "x=" from the equation
        if setName == "MATH":
            equ = d["equation"]
            equations = [equ[2:]]
            if useEqunSolutions:
                try:
                    targets = []
                except:
                    targets = []
                    # continue
            else:
                targets = ['disabled'] 
            print(targets)

        elif setName == "PEN":
            equations = d["equations"]
            mapNums = {}
            tempVars = ['X', 'Y', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            for k,v in d['answers'][0].items():
                mapNums[k] = tempVars.pop(0)
            for number in d['numbers']:
                replacements = [" ", "%", '-inch', '-dollar', '-point', '-pound', '-foot', '-feet', '-mile', '-yard', '-ounce', '-pint', '-quart', '-gallon', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', "ths", "-cent", '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', '-liter', '-ton', '-pound', '-inch', '-foot', '-yard', '-mile', '-hour', '-minute', '-second', '-week', '-month', '-year', '-day', '-century', '-decade', '-millennium', '-score', '-dozen', '-gross', '-acre', '-cent', '-kilometer', '-meter', '-gram', "-kilogram", '-peso', "-gallon", "gallon", "-page", "-legged", "-old"]
                temp = number['token'][0].lower()
                for r in replacements:
                    temp = temp.replace(r, "")

                mapNums[number['key']] = temp

            finalEquations = []
            for equation in equations:
                finalEqu = []
                splitEqu = equation.split(" ")
                for token in splitEqu:
                    if token in mapNums:
                        finalEqu.append(mapNums[token])
                    else:
                        finalEqu.append(token)
                almost = "".join(finalEqu).lower()
                for k,v in replace.items():
                    almost = almost.replace(k, str(v))
                almost = almost.replace("1-4th", "0.25")
                almost = almost.replace("4th", "0.25")
                finalEquations.append(almost)
            equations = finalEquations
            # equTemps = d["oldFormula"]
            # if type(equTemps) == str:
            #     equTemps = [equTemps]
            # equations = []
            # for equation in equTemps:
            #     equations.append("".join([i for i in equation if i != " " and i != ""]))
            if useEqunSolutions:
                try:
                    targets = [round(float(d["oldAnswer"][0]))]
                except:
                    targets = []
                    # continue
            else:
                targets = ['disabled']
        else:
            equations = d["lEquations"]
            # spEqs = []
            if useEqunSolutions:
                targets = d['lSolutions']
                # try:
                #     for equ in equations:
                #         sympy_eq = sp.simplify("Eq(" + equ.replace("=", ",") + ")")
                #         spEqs.append(sympy_eq)   
                #     solved = solve(spEqs, dict=True)
                #     targets = [round(i) for i in list(solved[0].values())]
                #     act_solns = list(round(i) for i in d['lSolutions'])
                #     same = 0
                #     for i, equ in enumerate(targets):
                #         if equ in act_solns:
                #             same += 1
                #     if same != len(targets):
                #         continue
                # except:
                    # continue
            else:
                targets = ['disabled'] 






        for s in seg:
            # search if its a number
            pos = re.search(pattern, s)

            # if its a number (pos is not None and the start of the number is at the start of the string)
            if pos and pos.start() == 0:
                # appeend the captured number only (not surrounding text in the word)
                nums.append(s[pos.start(): pos.end()])
                # mask the number in the input sequence
                input_seq.append("NUM")
                # if there was trailing text after the num (ex "80km/h" -> "80") append text to seq (ex "km/h")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                # not number: just append word
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        # fractions in the text
        nums_fraction = []

        # for nums in this problem
        for num in nums:
            # capture it if it's a fraction
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)

        # sort the fractions by length (not magnitude?). longest first
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        # seg the equation and tag the num
        # ex st: '(11-1)*2'
        def seg_and_tag(st):  
            # will become: 

            res = []
            # for largest to smallest fractions:
            for n in nums_fraction:
                # if fraction in this equation
                if n in st:
                    # find where in the equation
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    # if there is text before the fraction, seq_and_tag it seperately
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    # if this fraction is in the input text, append it as "N#"
                    if nums.count(n) == 1:
                        # pairNumMapping[n] = "N"+str(nums.index(n))
                        pairNumMapping["N"+str(nums.index(n))] = n
                        res.append("N"+str(nums.index(n)))
                    # if not, leave as variable
                    else:
                        res.append(n)
                    # recurse if text after number
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            # if no fractions, or fractions are not in equation 
            # sequence and tag non fractions
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            # if have number
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    # seq and tag text before number
                    res += seg_and_tag(st[:p_start])
                # strip text around number
                st_num = st[p_start:p_end]
                if st_num[-2:] == ".0":
                    st_num = st_num[:-2]
                if nums.count(st_num) > 0:
                    # same as fractions, append as "N#" if in the input text 
                    # pairNumMapping[st_num] = "N"+str(nums.index(st_num))
                    pairNumMapping["N"+str(nums.index(st_num))] = st_num
                    res.append("N"+str(nums.index(st_num)))
                else:
                    # if 
                    res.append(st_num)
                if p_end < len(st):
                    # seq and tag text after number
                    res += seg_and_tag(st[p_end:])
                return res
            # if no number
            for ss in st:
                # just keep text
                res.append(ss)
            return res

        # replace variables with X, Y, Z to be consistent
        varPattern = r'((x)|(y)|(z)|(a)|(b)|(c)|(d)|(e)|(f)|(g)|(h)|(i)|(j)|(k)|(n)|(m)|(o)|(p)|(q)|(r)|(s)|(t)|(u)|(v)|(w))'
        if setName == "MATH":
            allVars = ["X"]
        else:
            allVars = []
        allVarsMappings = []
        for eq in equations:
            matches = re.findall(varPattern, eq)
            allVarsTemp = []
            for match in matches:
                unique = list(set(match))
                allVarsTemp += unique
            allVars.append([item for item in list(set(allVarsTemp)) if item != "" ])
        allVarsParsed = list(set([var for vars in allVars for var in vars if var != ""]))
        allVarsTranslated = [variableHierarchy[i] for i in range(len(list(set([var for vars in allVars for var in vars if var != ""]))))]
        allVarsMappings = [{"var": var, "mapping": variableHierarchy[i]} for i, var in enumerate(allVarsParsed)]
        vars += allVarsTranslated

        newEquations = []
        for eq in equations:
            for mapping in allVarsMappings:
                eq = eq.replace(mapping["var"], mapping["mapping"])
            newEquations.append(eq)

        if setName == "MATH":
            allVars = ["X"]
        else:
            allVars = list([item['mapping'] for item in allVarsMappings])

        # tag the equation (replace numbers (only ones that are in the input text), in the equation with "N#")
        # ex: ['(', 'N1', '-', '1', ')', '*', 'N0']
        # out_seq = [seg_and_tag(equ) for equ in newEquations]
        print('newEquations', newEquations)
        out_seq = [seg_and_tag(equ) for equ in newEquations]

        # for each elem in equation sequence 
        for equ in out_seq:  
            for s in equ:
                # if the first char is a digit and it's not in the input text 
                # this happens if we have a number in the equation that is not in the input text
                # store 
                #   list of numbers in the equation that are not in the input text
                #   dict of the number and the number of times it appears in the equation
                if s[0].isdigit() and s not in generate_nums and s not in nums:
                    generate_nums.append(s)
                    generate_nums_dict[s] = 0
                if s in generate_nums and s not in nums:
                    generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)

        final_out_seq_list = []
        equationTargetVars = []
        if setName == "MATH":
            equationTargetVars = ['X']
            final_out_seq_list = out_seq
        else:
            for outputEquation in out_seq:
                # only want equations in this form
                if outputEquation[-2] == "=":
                    # if outputEquation[-1] == "375":
                    #     print()
                    equationTargetVars.append(outputEquation[-1])
                    final_out_seq_list.append(outputEquation[:-2])
                elif outputEquation[1] == "=":
                    # if outputEquation[0] == "375":
                    #     print()
                    equationTargetVars.append(outputEquation[0])
                    final_out_seq_list.append(outputEquation[2:])

                else:
                    continue
            if len(equationTargetVars) != len(out_seq):
                continue
        # input_seq: masked text
        # out_seq: equation with in text numbers replaced with "N#", and other numbers left as is
        # nums: list of numbers in the text
        # num_pos: list of positions of the numbers in the text
        # if "14" in equationTargetVars:
        #     print()
        # pairs.append((input_seq, final_out_seq_list, nums, num_pos, allVars, equationTargetVars, targets, pairNumMapping))
        pairs.append({
            "input_seq": input_seq,
            "equations": final_out_seq_list,
            "nums": nums,
            "num_pos": num_pos,
            "allVars": allVars,
            "equationTargetVars": equationTargetVars,
            "solution": targets,
            "pairNumMapping": pairNumMapping
        })

    temp_g = []
    for g in generate_nums:
        # only keep generated numbers if they are common in the text
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)

    # copy_nums: max length of numbers
    if useCustom:
        return pairs, temp_g, copy_nums, list(set(vars)) 
    else:
        return pairs, temp_g, copy_nums, [] 


def transfer_english_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs.append((input_seq, eq_segs, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


def transfer_roth_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = {}
    generate_nums = {}
    copy_nums = 0
    for key in data:
        d = data[key]
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"]

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                # if num[-2:] == ".0":
                #     num = num[:-2]
                # if "." in num and num[-1] == "0":
                #     num = num[:-1]
                nums.append(num.replace(",", ""))
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)
        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e not in "()+-*/":
                temp_eq += e
            elif temp_eq != "":
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq
                if len(count_eq) == 0:
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N"+str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # def seg_and_tag(st):  # seg the equation and tag the num
        #     res = []
        #     pos_st = re.search(pattern, st)
        #     if pos_st:
        #         p_start = pos_st.start()
        #         p_end = pos_st.end()
        #         if p_start > 0:
        #             res += seg_and_tag(st[:p_start])
        #         st_num = st[p_start:p_end]
        #         if st_num[-2:] == ".0":
        #             st_num = st_num[:-2]
        #         if "." in st_num and st_num[-1] == "0":
        #             st_num = st_num[:-1]
        #         if nums.count(st_num) == 1:
        #             res.append("N"+str(nums.index(st_num)))
        #         else:
        #             res.append(st_num)
        #         if p_end < len(st):
        #             res += seg_and_tag(st[p_end:])
        #     else:
        #         for sst in st:
        #             res.append(sst)
        #     return res
        # out_seq = seg_and_tag(equations)

        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        if len(nums) != 0:
            pairs[key] = (input_seq, eq_segs, nums, num_pos)

    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:
            temp_g.append(g)

    return pairs, temp_g, copy_nums


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, tree=False):
    res = []
    
    for word in sentence:
        # if word == "UNK":
        #     print()
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, vars, useCustom, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        # if not tree or pair[-1]:
        # add input sentence to vocab
        input_lang.add_sen_to_vocab(pair['input_seq'])
        # vocab for the equations. note that this does not add numbers or num tokens
        # to the lang
        for equ in pair['equations']:
            output_lang.add_sen_to_vocab(equ)
        # for equ in pair[1]:
        #     if equ == "+":
        #         print()
        # add outputs to lang
        # for tok in pair['equationTargetVars']:
        if useCustom:
            output_lang.add_sen_to_vocab(pair['equationTargetVars'])
    # this is hard coded at 5
    # cuts off words that appear less than 5 times 
    input_lang.build_input_lang(trim_min_count)


    # remove the variable tokens. we want to control where they go
    if useCustom:
        for var in vars:
            output_lang.remove_token_from_vocab(var)

    # add vars to input lang so we can pass them through encoder
    # input_lang.add_sen_to_vocab(vars)
    # hard coded to true
    if tree:
        # lang is the current lang + generate_nums array + N{i} (for each copy_num)
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums, vars, useCustom)
    else:
        # same but has pad, eos, unk tokens
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stacks = []
        # for word in the (masked) prefixed equation, if it's a number (so a constant), 
        # and not in the output lang, and in the list of numbers that are in the input text,
        # then store the locations of where in the nums list that number is

        # if its not in the nums list, then have that val in equation come from ALL
        # the numbers
        for equ in pair['equations']:
            num_stack = []
            for word in equ:
                temp_num = []
                flag_not = True
                # we already added equation to output lang, but numbers were not added
                # so capture the indexs of the constants
                if word not in output_lang.index2word:
                    flag_not = False
                    # for each in nums list
                    for i, j in enumerate(pair['nums']):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    # num_stack has the locations in the list of nums of where there is a number
                    # that is in the input text and the equation
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    # if no nums in both, let all numbers be in both??
                    num_stack.append([_ for _ in range(len(pair['nums']))])

            # ???
            num_stack.reverse()
            num_stacks.append(num_stack)

        # convert input sentence and equation into the vocab tokens
        input_cell = indexes_from_sentence(input_lang, pair['input_seq'])
        output_cell = [indexes_from_sentence(output_lang, equ, tree) for equ in pair['equations']]
        if useCustom:
            # equation_target = [output_lang.word2index[equ] for equ in pair['equationTargetVars']]
            equation_target = indexes_from_sentence(output_lang, pair['equationTargetVars'])
        else: 
            equation_target = pair['equationTargetVars']
        # equation_target = ["" for equ in pair[5]]
        # pair
        #   input: sentence with all numbers masked as NUM
        #   length of input
        #   output: prefix equation. Numbers from input as N{i}, non input as constants
        #   length of output
        #   nums: numbers from the input text
        #   loc nums: where nums are in the text
        #   [[] of where each token in the equation is found in the nums array]
        # train_pairs.append((input_cell, len(input_cell), output_cell, [len(equ) for equ in output_cell],
        #                     pair[2], pair[3], num_stacks, pair[4], equation_target, pair[6], pair[7]))
        train_pairs.append({
            "input_cell": input_cell,
            "input_len": len(input_cell),
            "equations" : output_cell,
            "equation_lens": [len(equ) for equ in output_cell],
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stacks,
            "allVars": pair['allVars'],
            "equationTargetVars": equation_target,
            "solution":  pair['solution'],
            "pairNumMapping": pair['pairNumMapping']
        })
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stacks = []
        for equ in pair['equations']:
            num_stack = []
            for word in equ: 
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair['nums']):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair['nums']))])

            num_stack.reverse()
            num_stacks.append(num_stack)
        input_cell = indexes_from_sentence(input_lang, pair['input_seq'])
        output_cell = [indexes_from_sentence(output_lang, equ, tree) for equ in pair['equations']]
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        # test_pairs.append((input_cell, len(input_cell), output_cell, [len(equ) for equ in output_cell],
        #                    pair[2], pair[3], num_stacks, pair[4], pair[5]))
        test_pairs.append({
            "input_cell": input_cell,
            "input_len": len(input_cell),
            "equations": output_cell,
            "equation_lens": [len(equ) for equ in output_cell],
            "nums": pair['nums'],
            "num_pos": pair['num_pos'],
            "num_stack": num_stacks,
            "allVars": pair['allVars'],
            "equationTargetVars": pair['equationTargetVars'],
            "solution": pair['solution'],
            "pairNumMapping": pair['pairNumMapping'],
        })
    print('Number of testind data %d' % (len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


def prepare_de_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        input_lang.add_sen_to_vocab(pair[0])
        output_lang.add_sen_to_vocab(pair[1])

    input_lang.build_input_lang(trim_min_count)

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        # train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack, pair[4]])
        train_pairs.append([input_cell, len(input_cell), pair[1], 0, pair[2], pair[3], num_stack])
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    for pair in pairs_tested:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        test_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
                           pair[2], pair[3], num_stack))
    print('Number of testind data %d' % (len(test_pairs)))
    # the following is to test out_equation
    # counter = 0
    # for pdx, p in enumerate(train_pairs):
    #     temp_out = allocation(p[2], 0.8)
    #     x = out_equation(p[2], p[4])
    #     y = out_equation(temp_out, p[4])
    #     if x != y:
    #         counter += 1
    #     ans = p[7]
    #     if ans[-1] == '%':
    #         ans = ans[:-1] + "/100"
    #     if "(" in ans:
    #         for idx, i in enumerate(ans):
    #             if i != "(":
    #                 continue
    #             else:
    #                 break
    #         ans = ans[:idx] + "+" + ans[idx:]
    #     try:
    #         if abs(eval(y + "-(" + x + ")")) < 1e-4:
    #             z = 1
    #         else:
    #             print(pdx, x, p[2], y, temp_out, eval(x), eval("(" + ans + ")"))
    #     except:
    #         print(pdx, x, p[2], y, temp_out, p[7])
    # print(counter)
    return input_lang, output_lang, train_pairs, test_pairs


# Pad a with the PAD symbol
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


# prepare the batches
def prepare_train_batch(pairs_to_batch, batch_size, vars, output_lang, input_lang):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    total_output_solutions = []
    total_targets = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    output_vars_batches = []

    var_pos_in_input = []
    var_size_in_input = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp['input_len'], reverse=True)
        input_length = []
        output_length = []
        output_vars = []
        output_var_solutions = []
        max_equ_length = 0
        # for each item in a pair
        input_len_max = 0
        targets = []
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        # get max number of eqautions in the batch:
        max_num_equ = 0
        input_len_max = 0
        targets_len_max = 0
        var_pos_in_inputs = []

        # for i, li, j, lj, num, num_pos, num_stack, var_list, equ_targets, var_solns, num_mapping in batch:
        for pair in batch:
            max_num_equ = max(max_num_equ, len(pair['equations']))
            max_equ_length = max(max_equ_length, max(pair['equation_lens']))
            # input_len_max = max(input_len_max, li + len(vars))
            input_len_max = max(input_len_max, pair['input_len'])
            targets_len_max = max(targets_len_max, len(pair['equationTargetVars']))

        # for i, li, j, lj, num, num_pos, num_stack, var_list, equ_targets, var_solns, num_mapping in batch:
        for pair in batch:
            # i = length if input in pair
            input_length.append(pair['input_len'])
            # j = length of output in pair
            output_length.append(pair['equation_lens'] + [0] * (max_num_equ - len(pair['equation_lens'])))
            
            # max_equ_length = max(max_equ_length, max(lj))

            num_batch.append(len(pair['nums']))
            # input batch: padded input text
            # inputs_with_vars_appended = i + [input_lang.word2index[i] for i in vars]
            # input_batch.append(pad_seq(inputs_with_vars_appended, li + len(vars), input_len_max))
            input_batch.append(pad_seq(pair['input_cell'], pair['input_len'], input_len_max))

            # var_pos = [li + i for i in range(len(var_list))]
            var_pos = [pair['input_len'] + i for i in range(len(pair['allVars']))]
            var_size = len(pair['allVars'])
            var_pos_in_inputs.append(var_pos)



            # output batch: padded output text
            output_temp = [pad_seq(equ, le, max_equ_length) for equ, le in zip(pair['equations'], pair['equation_lens'])]
            output_batch.append(output_temp + [pad_seq([], 0, max_equ_length) for _ in range(max_num_equ - len(pair['equations']))])
            # the corresponding arrays
            num_stack_batch.append(pair['num_stack'] + [[] for _ in range(max_num_equ - len(pair['num_stack']))])
            # positions of numbers
            num_pos_batch.append(pair['num_pos'])
            # size of numbers from input
            num_size_batch.append(len(pair['nums']))
            output_var_solutions.append(pair['solution'])
            targets.append(pair['equationTargetVars'] + [0 for _ in range(targets_len_max - len(pair['equationTargetVars']))])

            cur_vars = []
            for var in vars:
                if var in pair['allVars']:
                    cur_vars.append(0)
                else:
                    cur_vars.append(1)
            output_vars.append(cur_vars)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        output_batches.append(output_batch)
        output_vars_batches.append(output_vars)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        total_output_solutions.append(output_var_solutions)
        total_targets.append(targets)
        var_pos_in_input.append(var_pos_in_inputs)
    # input_batches: padded inputs
    # input_lengths: length of the inputs (without padding)
    # output_batches: padded outputs
    # output_length: length of the outputs (without padding)
    # num_batches: numbers from the input text 
    # num_stack_batches: the corresponding nums lists
    # num_pos_batches: positions of the numbers lists
    # num_size_batches: number of numbers from the input text
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, output_vars_batches, total_output_solutions, total_targets, var_pos_in_input


def get_num_stack(eq, output_lang, num_pos):
    num_stack = []
    for word in eq:
        temp_num = []
        flag_not = True
        if word not in output_lang.index2word:
            flag_not = False
            for i, j in enumerate(num_pos):
                if j == word:
                    temp_num.append(i)
        if not flag_not and len(temp_num) != 0:
            num_stack.append(temp_num)
        if not flag_not and len(temp_num) == 0:
            num_stack.append([_ for _ in range(len(num_pos))])
    num_stack.reverse()
    return num_stack


def prepare_de_train_batch(pairs_to_batch, batch_size, output_lang, rate, english=False):
    pairs = []
    b_pairs = copy.deepcopy(pairs_to_batch)
    for pair in b_pairs:
        p = copy.deepcopy(pair)
        pair[2] = check_bracket(pair[2], english)

        temp_out = exchange(pair[2], rate)
        temp_out = check_bracket(temp_out, english)

        p[2] = indexes_from_sentence(output_lang, pair[2])
        p[3] = len(p[2])
        pairs.append(p)

        temp_out_a = allocation(pair[2], rate)
        temp_out_a = check_bracket(temp_out_a, english)

        if temp_out_a != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out_a, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out_a)
            p[3] = len(p[2])
            pairs.append(p)

        if temp_out != pair[2]:
            p = copy.deepcopy(pair)
            p[6] = get_num_stack(temp_out, output_lang, p[4])
            p[2] = indexes_from_sentence(output_lang, temp_out)
            p[3] = len(p[2])
            pairs.append(p)

            if temp_out_a != pair[2]:
                p = copy.deepcopy(pair)
                temp_out_a = allocation(temp_out, rate)
                temp_out_a = check_bracket(temp_out_a, english)
                if temp_out_a != temp_out:
                    p[6] = get_num_stack(temp_out_a, output_lang, p[4])
                    p[2] = indexes_from_sentence(output_lang, temp_out_a)
                    p[3] = len(p[2])
                    pairs.append(p)
    print("this epoch training data is", len(pairs))
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        for i, li, j, lj, num, num_pos, num_stack in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches


# Multiplication exchange rate
def exchange(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    while idx < len(ex):
        s = ex[idx]
        if (s == "*" or s == "+") and random.random() < rate:
            lidx = idx - 1
            ridx = idx + 1
            if s == "+":
                flag = 0
                while not (lidx == -1 or ((ex[lidx] == "+" or ex[lidx] == "-") and flag == 0) or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex) or ((ex[ridx] == "+" or ex[ridx] == "-") and flag == 0) or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            else:
                flag = 0
                while not (lidx == -1
                           or ((ex[lidx] == "+" or ex[lidx] == "-" or ex[lidx] == "*" or ex[lidx] == "/") and flag == 0)
                           or flag == 1):
                    if ex[lidx] == ")" or ex[lidx] == "]":
                        flag -= 1
                    elif ex[lidx] == "(" or ex[lidx] == "[":
                        flag += 1
                    lidx -= 1
                if flag == 1:
                    lidx += 2
                else:
                    lidx += 1

                flag = 0
                while not (ridx == len(ex)
                           or ((ex[ridx] == "+" or ex[ridx] == "-" or ex[ridx] == "*" or ex[ridx] == "/") and flag == 0)
                           or flag == -1):
                    if ex[ridx] == ")" or ex[ridx] == "]":
                        flag -= 1
                    elif ex[ridx] == "(" or ex[ridx] == "[":
                        flag += 1
                    ridx += 1
                if flag == -1:
                    ridx -= 2
                else:
                    ridx -= 1
            if lidx > 0 and ((s == "+" and ex[lidx - 1] == "-") or (s == "*" and ex[lidx - 1] == "/")):
                lidx -= 1
                ex = ex[:lidx] + ex[idx:ridx + 1] + ex[lidx:idx] + ex[ridx + 1:]
            else:
                ex = ex[:lidx] + ex[idx + 1:ridx + 1] + [s] + ex[lidx:idx] + ex[ridx + 1:]
            idx = ridx
        idx += 1
    return ex


def check_bracket(x, english=False):
    if english:
        for idx, s in enumerate(x):
            if s == '[':
                x[idx] = '('
            elif s == '}':
                x[idx] = ')'
        s = x[0]
        idx = 0
        if s == "(":
            flag = 1
            temp_idx = idx + 1
            while flag > 0 and temp_idx < len(x):
                if x[temp_idx] == ")":
                    flag -= 1
                elif x[temp_idx] == "(":
                    flag += 1
                temp_idx += 1
            if temp_idx == len(x):
                x = x[idx + 1:temp_idx - 1]
            elif x[temp_idx] != "*" and x[temp_idx] != "/":
                x = x[idx + 1:temp_idx - 1] + x[temp_idx:]
        while True:
            y = len(x)
            for idx, s in enumerate(x):
                if s == "+" and idx + 1 < len(x) and x[idx + 1] == "(":
                    flag = 1
                    temp_idx = idx + 2
                    while flag > 0 and temp_idx < len(x):
                        if x[temp_idx] == ")":
                            flag -= 1
                        elif x[temp_idx] == "(":
                            flag += 1
                        temp_idx += 1
                    if temp_idx == len(x):
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1]
                        break
                    elif x[temp_idx] != "*" and x[temp_idx] != "/":
                        x = x[:idx + 1] + x[idx + 2:temp_idx - 1] + x[temp_idx:]
                        break
            if y == len(x):
                break
        return x

    lx = len(x)
    for idx, s in enumerate(x):
        if s == "[":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == "]":
                    flag_b += 1
                elif x[temp_idx] == "[":
                    flag_b -= 1
                if x[temp_idx] == "(" or x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == "]" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "("
                x[temp_idx] = ")"
                continue
        if s == "(":
            flag_b = 0
            flag = False
            temp_idx = idx
            while temp_idx < lx:
                if x[temp_idx] == ")":
                    flag_b += 1
                elif x[temp_idx] == "(":
                    flag_b -= 1
                if x[temp_idx] == "[":
                    flag = True
                if x[temp_idx] == ")" and flag_b == 0:
                    break
                temp_idx += 1
            if not flag:
                x[idx] = "["
                x[temp_idx] = "]"
    return x


# Multiplication allocation rate
def allocation(ex_copy, rate):
    ex = copy.deepcopy(ex_copy)
    idx = 1
    lex = len(ex)
    while idx < len(ex):
        if (ex[idx] == "/" or ex[idx] == "*") and (ex[idx - 1] == "]" or ex[idx - 1] == ")"):
            ridx = idx + 1
            r_allo = []
            r_last = []
            flag = 0
            flag_mmd = False
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag += 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        r_last = ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                    elif ex[ridx] == "*" or ex[ridx] == "/":
                        flag_mmd = True
                        r_last = [")"] + ex[ridx:]
                        r_allo = ex[idx + 1: ridx]
                        break
                elif flag == -1:
                    r_last = ex[ridx:]
                    r_allo = ex[idx + 1: ridx]
                    break
                ridx += 1
            if len(r_allo) == 0:
                r_allo = ex[idx + 1:]
            flag = 0
            lidx = idx - 1
            flag_al = False
            flag_md = False
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag -= 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[lidx] == "+" or ex[lidx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                lidx -= 1
            if lidx != 0 and ex[lidx - 1] == "/":
                flag_al = False
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = lidx + 1
                temp_res = ex[:lidx]
                if flag_mmd:
                    temp_res += ["("]
                if lidx - 1 > 0:
                    if ex[lidx - 1] == "-" or ex[lidx - 1] == "*" or ex[lidx - 1] == "/":
                        flag_md = True
                        temp_res += ["("]
                flag = 0
                lidx += 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 0:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    temp_idx += 1
                temp_res += ex[lidx: temp_idx] + [ex[idx]] + r_allo
                if flag_md:
                    temp_res += [")"]
                temp_res += r_last
                return temp_res
        if ex[idx] == "*" and (ex[idx + 1] == "[" or ex[idx + 1] == "("):
            lidx = idx - 1
            l_allo = []
            temp_res = []
            flag = 0
            flag_md = False  # flag for x or /
            while lidx > 0:
                if ex[lidx] == "(" or ex[lidx] == "[":
                    flag += 1
                elif ex[lidx] == ")" or ex[lidx] == "]":
                    flag -= 1
                if flag == 0:
                    if ex[lidx] == "+":
                        temp_res = ex[:lidx + 1]
                        l_allo = ex[lidx + 1: idx]
                        break
                    elif ex[lidx] == "-":
                        flag_md = True  # flag for -
                        temp_res = ex[:lidx] + ["("]
                        l_allo = ex[lidx + 1: idx]
                        break
                elif flag == 1:
                    temp_res = ex[:lidx + 1]
                    l_allo = ex[lidx + 1: idx]
                    break
                lidx -= 1
            if len(l_allo) == 0:
                l_allo = ex[:idx]
            flag = 0
            ridx = idx + 1
            flag_al = False
            all_res = []
            while ridx < lex:
                if ex[ridx] == "(" or ex[ridx] == "[":
                    flag -= 1
                elif ex[ridx] == ")" or ex[ridx] == "]":
                    flag += 1
                if flag == 1:
                    if ex[ridx] == "+" or ex[ridx] == "-":
                        flag_al = True
                if flag == 0:
                    break
                ridx += 1
            if not flag_al:
                idx += 1
                continue
            elif random.random() < rate:
                temp_idx = idx + 1
                flag = 0
                lidx = temp_idx + 1
                while temp_idx < idx - 1:
                    if ex[temp_idx] == "(" or ex[temp_idx] == "[":
                        flag -= 1
                    elif ex[temp_idx] == ")" or ex[temp_idx] == "]":
                        flag += 1
                    if flag == 1:
                        if ex[temp_idx] == "+" or ex[temp_idx] == "-":
                            all_res += l_allo + [ex[idx]] + ex[lidx: temp_idx] + [ex[temp_idx]]
                            lidx = temp_idx + 1
                    if flag == 0:
                        break
                    temp_idx += 1
                if flag_md:
                    temp_res += all_res + [")"]
                elif ex[temp_idx + 1] == "*" or ex[temp_idx + 1] == "/":
                    temp_res += ["("] + all_res + [")"]
                temp_res += ex[temp_idx + 1:]
                return temp_res
        idx += 1
    return ex


