from typing import List

class FenwickTree():
    def __init__(self, n: int, arr: List[int] = None):
        """
        Initialize fenwick tree for an array of size n
        :param n: Size of array
        :param arr: Pre-initialization of array. Almost all use-cases wouldn't require this argument.
        """
        self.BITTree = self.construct(n)
        self.n = n

    # Returns sum of arr[0..index]. This function assumes
    # that the array is preprocessed and partial sums of
    # array elements are stored in BITree[].
    def getsum(self, i):
        s = 0  # initialize result

        # index in BITree[] is 1 more than the index in arr[]
        i = i + 1

        # Traverse ancestors of BITree[index]
        while i > 0:
            # Add current element of BITree to sum
            s += self.BITTree[i]

            # Move index to parent node in getSum View
            i -= i & (-i)
        return s


    # Updates a node in Binary Index Tree (BITree) at given index
    # in BITree. The given value 'val' is added to BITree[i] and
    # all of its ancestors in tree.
    def updatebit(self, i, v):
        # index in BITree[] is 1 more than the index in arr[]
        i += 1

        # Traverse all ancestors and add 'val'
        while i <= self.n:
            # Add 'val' to current node of BI Tree
            self.BITTree[i] += v

            # Update index to that of parent in update View
            i += i & (-i)


    # Constructs and returns a Binary Indexed Tree for given
    # array of size n.
    def construct(self, n, arr=None):
        # Create and initialize BITree[] as 0
        BITTree = [0] * (n + 1)
        # # Store the actual values in BITree[] using update()
        # for i in range(n):
        #     self.updatebit(BITTree, n, i, arr[i])

        # Uncomment below lines to see contents of BITree[]
        # for i in range(1,n+1):
        #     print BITTree[i],
        return BITTree

    def getlistidx(self, idx):
        return idx + self.getsum(idx)


# class IndexUpdatingList


if __name__=='__main__':
    n = 10
    fwtree = FenwickTree(n)
    fwtree.updatebit(i=2, v=1)
    print(fwtree.getlistidx(2))

    fwtree.updatebit(i=4, v=1)
    print(fwtree.getlistidx(3))
    print(fwtree.getlistidx(4))
    print(fwtree.getlistidx(2))