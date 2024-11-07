import matplotlib.pyplot as plt


# data = [[108499996508160.0, 108249992921088.0, 108249999212544.0, 110000001253376.0, 106750000758784.0, 109000003682304.0, 109750001860608.0, 105749996896256.0, 107750000427008.0, 107249999544320.0, 107999999819776.0, 106249999876096.0, 105750003187712.0, 105000000815104.0, 106249999876096.0, 107749998329856.0, 105250000207872.0, 107499998937088.0, 107249997447168.0, 106749998661632.0, 107250001641472.0, 108499998605312.0, 103249996677120.0, 105999998386176.0, 106000000483328.0, 106499997171712.0, 107500001034240.0, 108749997998080.0, 110000005447680.0, 109000003682304.0, 107250003738624.0, 107749998329856.0, 108500002799616.0, 107999995625472.0, 108500002799616.0, 109499998273536.0, 107999999819776.0, 105750001090560.0, 105249998110720.0, 108000001916928.0, 107749998329856.0, 107250003738624.0, 108250003406848.0, 104749999325184.0, 107249997447168.0, 109250000977920.0, 106249999876096.0, 108749997998080.0, 106749998661632.0, 108500000702464.0, 109249998880768.0, 108000001916928.0, 108000001916928.0, 105999998386176.0, 107250001641472.0, 106999998054400.0, 107250001641472.0, 108250001309696.0, 105750001090560.0, 106500001366016.0, 107750004621312.0, 107750002524160.0, 107499998937088.0, 108749995900928.0, 106999998054400.0, 107250003738624.0, 109250000977920.0, 107249993252864.0, 107750002524160.0, 107750000427008.0, 106500001366016.0, 104000001146880.0, 106249999876096.0, 103250002968576.0, 106499997171712.0, 106000002580480.0, 108500002799616.0, 105999998386176.0, 108000006111232.0, 110500000038912.0]]
# data = [[3258.8988037109375, 2654.1727294921875, 2545.98095703125, 2040.6771240234375, 1970.4481811523438, 1804.6610717773438, 1916.6027221679688, 1714.1339721679688, 1655.3868408203125, 1712.4122924804688, 1652.6566772460938, 1526.2322387695312, 1522.0205078125, 1570.0613403320312, 1497.4074096679688, 1468.4285888671875, 1421.2301025390625, 1389.4251708984375, 1378.4092407226562, 1349.0286865234375, 1355.9716186523438, 1361.1325073242188, 1303.7977294921875, 1253.1943969726562, 1285.9205322265625, 1285.4847412109375, 1293.4033813476562, 1280.6954345703125, 1243.036376953125, 1207.0443725585938, 1225.7804565429688, 1264.8792114257812, 1184.393310546875, 1159.251220703125, 1197.6041870117188, 1172.421142578125, 1112.78515625, 1164.0597534179688, 1144.4221801757812, 1141.70263671875, 1155.235107421875, 1099.5921020507812, 1104.8901977539062, 1093.7745971679688, 1072.5927124023438, 1071.7034301757812, 1048.5525512695312, 1039.5083312988281, 1097.1100463867188, 1048.8078002929688, 1020.78564453125, 1052.45166015625, 1085.67041015625, 1066.6453247070312, 1050.4952392578125, 998.8089294433594, 1037.8948059082031, 1040.9211120605469, 1021.6159973144531, 960.3216247558594, 1018.5894775390625, 1015.961181640625, 1013.8593444824219, 991.7834777832031, 1005.5584106445312, 960.757568359375, 1006.8442687988281, 995.3585815429688, 996.4570617675781, 976.8101196289062, 967.7835083007812, 943.5509948730469, 992.5048217773438, 993.0097045898438, 943.5267028808594, 947.2035217285156, 972.9401550292969, 947.8102416992188, 943.7104187011719, 989.2584838867188]]
data = [[0.014796375589838262, 0.20642325002190265, 0.01645934676500456, 0.175531724771323, 0.22755436476414578, 0.22820157773872476, 0.2384292228955703, 0.23327698050672363, 0.23018987818438905, 0.23810778655704115, 0.22958854929474348, 0.24399498927039331, 0.2574281698728093, 0.23455889032958463, 0.23943758644338775, 0.22567742649022682, 0.2379333752005451, 0.23825275536085835, 0.24424516168200275, 0.25706105433321463, 0.2955599323430409, 0.27511098733605166, 0.2483241775725089, 0.23393000459434257, 0.24592555110177222, 0.25888950499455166, 0.270438903066313, 0.27144205824325657, 0.2699499301632674, 0.24032392250282608, 0.21514721778336035, 0.2511628606632038, 0.25302148180696005, 0.24690966045716234, 0.22583438255823018, 0.24877517314919884, 0.23847431659545715, 0.1950248686709294, 0.19193164745968688, 0.19848766915200702, 0.18301693392517426, 0.1833132230019511, 0.18237881546783052, 0.18743470012199345, 0.18545507604767786, 0.19776883718309715, 0.19503009587533326, 0.1972839109268528, 0.1978750095405326, 0.19034152181242192, 0.18874521682892967, 0.19460941244797525, 0.20433111472285576, 0.2137463736022778, 0.21401348234172146, 0.2183497486579017, 0.21436581529214557, 0.20101682631094397, 0.19546924142333041, 0.19883872315348833, 0.20646926725711584, 0.20831012367472979, 0.214335628745834, 0.22158632799079434, 0.21760968929392613, 0.21722254256531598, 0.2149371976937319, 0.20576528878007266, 0.1990260912854638, 0.20070676355437134, 0.20621575494072536, 0.21251822339527415, 0.21282156424380871, 0.21364910922705202, 0.21393605327296306, 0.21593860190641428, 0.21648814323070453, 0.21216800617025186, 0.21287340685569123, 0.21460754695924067]]
filename = 'loss_graph.png'
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss of mGTS on DRAW1k')

t = [i for i in range(len(data[0]))]
for i in range(len(data)):
    plt.plot(t, data[i])
# plt.plot(data)
# plt.plot(validationLoss)
# plt.figlegend(['train', 'validation'], loc='upper left')
# plt.figlegend([f'Fold {i}' for i in range(len(data))], loc='upper right')
# plt.figlegend(['train'], loc='upper left')
plt.show()
# plt.savefig(filename)