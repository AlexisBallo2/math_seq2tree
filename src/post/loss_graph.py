import matplotlib.pyplot as plt


# data = [[108499996508160.0, 108249992921088.0, 108249999212544.0, 110000001253376.0, 106750000758784.0, 109000003682304.0, 109750001860608.0, 105749996896256.0, 107750000427008.0, 107249999544320.0, 107999999819776.0, 106249999876096.0, 105750003187712.0, 105000000815104.0, 106249999876096.0, 107749998329856.0, 105250000207872.0, 107499998937088.0, 107249997447168.0, 106749998661632.0, 107250001641472.0, 108499998605312.0, 103249996677120.0, 105999998386176.0, 106000000483328.0, 106499997171712.0, 107500001034240.0, 108749997998080.0, 110000005447680.0, 109000003682304.0, 107250003738624.0, 107749998329856.0, 108500002799616.0, 107999995625472.0, 108500002799616.0, 109499998273536.0, 107999999819776.0, 105750001090560.0, 105249998110720.0, 108000001916928.0, 107749998329856.0, 107250003738624.0, 108250003406848.0, 104749999325184.0, 107249997447168.0, 109250000977920.0, 106249999876096.0, 108749997998080.0, 106749998661632.0, 108500000702464.0, 109249998880768.0, 108000001916928.0, 108000001916928.0, 105999998386176.0, 107250001641472.0, 106999998054400.0, 107250001641472.0, 108250001309696.0, 105750001090560.0, 106500001366016.0, 107750004621312.0, 107750002524160.0, 107499998937088.0, 108749995900928.0, 106999998054400.0, 107250003738624.0, 109250000977920.0, 107249993252864.0, 107750002524160.0, 107750000427008.0, 106500001366016.0, 104000001146880.0, 106249999876096.0, 103250002968576.0, 106499997171712.0, 106000002580480.0, 108500002799616.0, 105999998386176.0, 108000006111232.0, 110500000038912.0]]
# data = [[3258.8988037109375, 2654.1727294921875, 2545.98095703125, 2040.6771240234375, 1970.4481811523438, 1804.6610717773438, 1916.6027221679688, 1714.1339721679688, 1655.3868408203125, 1712.4122924804688, 1652.6566772460938, 1526.2322387695312, 1522.0205078125, 1570.0613403320312, 1497.4074096679688, 1468.4285888671875, 1421.2301025390625, 1389.4251708984375, 1378.4092407226562, 1349.0286865234375, 1355.9716186523438, 1361.1325073242188, 1303.7977294921875, 1253.1943969726562, 1285.9205322265625, 1285.4847412109375, 1293.4033813476562, 1280.6954345703125, 1243.036376953125, 1207.0443725585938, 1225.7804565429688, 1264.8792114257812, 1184.393310546875, 1159.251220703125, 1197.6041870117188, 1172.421142578125, 1112.78515625, 1164.0597534179688, 1144.4221801757812, 1141.70263671875, 1155.235107421875, 1099.5921020507812, 1104.8901977539062, 1093.7745971679688, 1072.5927124023438, 1071.7034301757812, 1048.5525512695312, 1039.5083312988281, 1097.1100463867188, 1048.8078002929688, 1020.78564453125, 1052.45166015625, 1085.67041015625, 1066.6453247070312, 1050.4952392578125, 998.8089294433594, 1037.8948059082031, 1040.9211120605469, 1021.6159973144531, 960.3216247558594, 1018.5894775390625, 1015.961181640625, 1013.8593444824219, 991.7834777832031, 1005.5584106445312, 960.757568359375, 1006.8442687988281, 995.3585815429688, 996.4570617675781, 976.8101196289062, 967.7835083007812, 943.5509948730469, 992.5048217773438, 993.0097045898438, 943.5267028808594, 947.2035217285156, 972.9401550292969, 947.8102416992188, 943.7104187011719, 989.2584838867188]]
data = [[0.014796375589838262, 0.2244740081687246, 0.014796375589838262, 0.16401610504910358, 0.2307056928821013, 0.21534060371250835, 0.22211102844438255, 0.2042811156503378, 0.21141076231900302, 0.2512398508138016, 0.22314971044245582, 0.2521336647535892, 0.24997946127507656, 0.24609981545980578, 0.21269664587798237, 0.18609569458674324, 0.2315951175710394, 0.2351655880213051, 0.2579610261445457, 0.23890706255187583, 0.23489194806330366, 0.22971504302987072, 0.23127810932751375, 0.24249211133061185, 0.24924515977787773, 0.25402976628165336, 0.25696136043275963, 0.23908435453816293, 0.24297470617600384, 0.25731034177668854, 0.2551866542927615, 0.24091154293761752, 0.22334517595324807, 0.23783216873229984, 0.2263269978103739, 0.2413464850773816, 0.23483033888186422, 0.21132337917628927, 0.20284769258008614, 0.1990151469971819, 0.19848940301376222, 0.20748962353915265, 0.20663505327281767, 0.20811903398204945, 0.20904281608230216, 0.2002650427982597, 0.19763175124974142, 0.19914447164712273, 0.19730327677991577, 0.19262488805561606, 0.20558374768654863, 0.2022392713435069, 0.19867748488109038, 0.19977198018056355, 0.19417915016318107, 0.19495570502868864, 0.19483985805363138, 0.1953487314188456, 0.2063554326307742, 0.21300179772265113, 0.2139040647318359, 0.21546465220471836, 0.2158330626879064, 0.21232400678286775, 0.2093984656033188, 0.20845382758550998, 0.20581433094314389, 0.20520805056641872, 0.19948628461759269, 0.20210167185028388, 0.20171255037077507, 0.20072902887075433, 0.20277150015407144, 0.20284002961877082, 0.2093790413801643, 0.2095158470262956, 0.20804095017119792, 0.2105248641391117, 0.21192390867956967, 0.2123827914196576]]
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