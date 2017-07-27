import random, time

def quick_sort(l):
    if len(l) <= 1:
        return l

    i = 0
    j = len(l) - 1

    while True:
        while l[j] >= l[0] and j > 0:
            j -= 1
        while l[i] <= l[0] and i < len(l) - 1 and i < j:
            i += 1
        if i != j:
            l[i], l[j] = l[j], l[i]
        else:
            l[0], l[j] = l[j], l[0]
            l = quick_sort(l[:j]) + [l[j]] + quick_sort(l[j+1:])
            return l

def insert_sort(l):
    rl = [l[0]]
    for i in range(1, len(l)):
        flag = True
        for i_rl in range(len(rl)):
            if rl[i_rl] > l[i]:
                rl.insert(i_rl, l[i])
                flag = False
                break
        if flag:
            rl.append(l[i])
    return rl

def new_quick_sort(l):
    if len(l) <= 5:
        return insert_sort(l)

    i = 0
    j = len(l) - 1

    while True:
        while l[j] >= l[0] and j > 0:
            j -= 1
        while l[i] <= l[0] and i < len(l) - 1 and i < j:
            i += 1
        if i != j:
            l[i], l[j] = l[j], l[i]
        else:
            l[0], l[j] = l[j], l[0]
            l = quick_sort(l[:j]) + [l[j]] + quick_sort(l[j+1:])
            return l


if __name__ == '__main__':
    random_list = [random.randint(0, 50000) for x in range(50000)]
    # print(random_list)
    random_list_1 = [x for x in random_list]
    random_list_2 = [x for x in random_list]
    random_list_3 = [x for x in random_list]
    # print(random_list)

    now = time.time()
    random_list_quick_sort = quick_sort(random_list)
    # print(random_list_quick_sort)
    print("快速排序", time.time() - now)

    # now = time.time()
    # random_list_quick_sort = insert_sort(random_list_1)
    # # print(random_list_quick_sort)
    # print("插入排序", time.time() - now)

    now = time.time()
    random_list_quick_sort = new_quick_sort(random_list_2)
    # print(random_list_quick_sort)
    print("新快速排序", time.time() - now)

    # now = time.time()
    # random_list_1.sort()
    # print(time.time() - now)
    # print(random_list_quick_sort)