def decodeString(s):
    """
    :type s: str
    :rtype: str
    """

    stack = []
    slist = list(s)
    res = ''

    while slist:
        digit = ''
        sstr = ''

        while slist and ('0' <= slist[0] <= '9'):
            digit += slist[0]
            del slist[0]

        if digit:
            stack.append(int(digit))

        while slist and (('a' <= slist[0] <= 'z') or ('A' <= slist[0] <= 'Z')):
            sstr += slist[0]
            del slist[0]
        if sstr:
            stack.append(sstr)

        if slist and slist[0] == '[':
            del slist[0]
        elif slist and slist[0] == ']':
            del slist[0]

            str1 = stack.pop()
            dig1 = stack.pop()

            if stack:
                str2 = stack.pop()
                stack.append(str2 + str1 * dig1)
            else:
                stack.append(str1 * dig1)

        if len(stack) == 1 and type(stack[0]) == str:
            res = res + stack.pop()

    return res


decodeString("3[a]2[b4[F]c]")

