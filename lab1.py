def goal(*args):
    
    result = True
    numbers = list(args)
    for i in range (len(numbers)):
        if numbers[0]!=numbers[i]:
            result = False
            break
    return result

        
def generateAndTest(list1, list2, list3, list4, list5, goal):
    for a in list1:
        for b in list2:
            for c in list3:
                for d in list4:
                    for e in list5:
                        if (goal(a,b,c,d,e)):
                            print("- POTENTIAL MEETING at {}:00".format(e))

Conference_room = [[8, 9, 13,15],[9,11,16],[8,9,10,12],[10,12,13,15,16],[8,9,10,11,12,13,14,15,16]]
Yourself = [[8, 12, 13, 15, 16],[8, 9, 11, 13, 14],[8, 9, 11, 13, 14],[8, 9, 11, 13, 14],[8, 9, 11, 13, 14, 15]]
Anna = [[9, 10, 11, 13, 15],[9, 10, 11],[8, 9, 10, 11, 13, 14, 15],[9, 10, 13],[8, 9, 11, 12, 15]]
Bob = [[10, 11, 13, 15],[8, 9, 10, 11],[10, 11, 13, 15],[8, 9, 13, 15],[8, 9, 11, 13]]
Carrie = [[8, 9, 10, 13, 14, 15],[11, 12, 13, 14, 15],[8, 9, 10, 13, 14, 15],[12, 13, 14, 15],[8, 9, 10, 11, 13]]

Schedule = [Conference_room, Yourself, Anna, Bob, Carrie]
days = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY']

for i in range(len(Schedule[1])):
    print(days[i],':')
    generateAndTest(Schedule[0][i], Schedule[1][i], Schedule[2][i], Schedule[3][i], Schedule[4][i], goal)
    print()