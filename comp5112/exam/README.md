# COMP5112 Exam

### Question 1
An SQL select statement

- [ ] must have the select clause
- [ ] must have the from clause
- [ ] None of the others
- [ ] must have the sort by clause
- [ ] must have the where clause

### Question 2
A queue data structure cannot have both qFront and qEnd equal to NULL.

- [ ] True
- [ ] False

### Question 3
Advantages of file processing over the database approach are

- [ ] program-data dependence
- [ ] no conversion cost
- [ ] None of the others
- [ ] no need for specialized personnel
- [ ] increased data sharing

### Question 4
The root node of a 8-tree with m keys in every node must at least have one key

- [ ] True
- [ ] False

### Question 5
Hashing with linear probe can

- [ ] always find an empty entry to add data
- [ ] have no collisions
- [ ] None of the others
- [ ] have one collision
- [ ] have more than one collision

### Question 6
Data for binary search must be stored in an array for log(n) time complexity instead of a linked list, where n is the number of items in the array.

- [ ] True
- [ ] False

### Question 7
Binary search is one of the indexing method in databases

- [ ] True
- [ ] False

### Question 8
The disadvantage of join index is the increased storage cost comQared with sequential search

- [ ] True
- [ ] False

### Question 9
Given a relation schema R = (A, B, C, D, E) with the set X of functional dependencies: A-> B, A->C, 8->D, D->E, $B^+$ is/are

- [ ] BDE
- [ ] DE
- [ ] BD
- [ ] B
- [ ] AB

### Question 10
The following expression, `*ab+c*bd`, is

- [ ] None of the others
- [ ] a postfix expression
- [ ] a prefix expression
- [ ] an infix expression

### Question 11
An item has many items and an item belongs to different items. In ER model, such a relationship is
- [ ] a unary relationship
- [ ] many-to-many relationship
- [ ] one-to-many relationship
- [ ] None of the others
- [ ] a binary relationship


### Question 12
The advantages of vertical partitioning in database are
- [ ] None of the others
- [ ] each partition can be optimized for performance
- [ ] easier to programme
- [ ] less bottleneck using different disks for different partitions
- [ ] records used together are grouped together


### Question 13
The worst-case time complexity of heap-increase-key is smaller than that of max-heapify.

- [ ] True
- [ ] False


### Question 14
A field must have a unique name in a relation

- [ ] True
- [ ] False

### Question 15
Given the following program fragment: for(i = 1; i < n; ++i) { for(j=0; j < n; ++j) if (data[j] > data[j-1]) swap(data,j,j-1 );}, this program fragment can crash where i, j, n, data and swap() are appropriately declared or defined.

- [ ] True
- [ ] False

### Question 16
Given a relation schema R = (A, B, C, D) with the set X of functional dependencies: A-> B, A->C, B->D, AD is a candidate key.

- [ ] True
- [ ] False

### Question 17
Given the declaration: struct node { int f; struct node* next};, the following program can crash: void Pop(int & f) {struct
node * t; if (top != NULL) {f = top->f; t = top; top = top->next; delete t;}} where top is the variable that points to the top of the stack.

- [ ] True
- [ ] False

### Question 18

In ER modeling, a weak entity does not have identifier attribute

- [ ] True
- [ ] False

### Question 19
Given the load factor is 0.4 for the current hash table size, if the hash table size is doubled with a new hash function, what is the new average number of table elements examined during successful search (rounded to the nearest 3 decimal places) for open addressing with double hashing?


### Question 20

The advantage of RAID-4 over RAID-3 is to access multiple records per stripe

- [ ] True
- [ ] False

### Question 21

In Business Rules, an anchor object can be a corresponding object

- [ ] True
- [ ] False

### Question 22
The worst-case time complexity of quicksort is the same as that of bubble sort.

- [ ] True
- [ ] False

### Question 23
In relational algebra, the union of two relations can be undefined

- [ ] True
- [ ] False

### Question 24
Given a relation schema R = (A, B, C) with the set F of functional dependencies: A-> B and A-> BC, the minimal cover of F is the same as F since there are no transitive dependencies.

- [ ] True
- [ ] False

### Question 25
In EER modeling, a subtype must have at least two attributes: an identifier attribute and a discriminator attribute.

- [ ] True
- [ ] False

### Question 26
Mergesort requires more storage than Quicksort.

- [ ] True
- [ ] False


### Question 27
Relations of a database are in third normal form if

- [ ] the relations are at least in second normal form
- [ ] the relations are at most in third normal form
- [ ] None of the others
- [ ] the relations are at least in Boyce-Codd normal form
- [ ] the relations are at least in fourth normal form


### Question 28
SQL is a query language that only queries a database and retrieve results as a table to the user

- [ ] True
- [ ] False

### Question 29
For the declaration: struct _node { int freq; struct _node* left; struct _node right; };, the declaration can:

- [ ] be data used in the program
- [ ] be a data structure
- [ ] be a parameter
- [ ] be for binary tree nodes
- [ ] None of the others.
- [ ] be for general tree nodes with any number of children nodes

### Question 30
Any data is metadata

- [ ] True
- [ ] False

### Question 31
The project operation/operator in relational algebra is the same as the Select clause of the SQL Select statement.

- [ ] True
- [ ] False

### Question 32
In ER modeling, an attribute cannot be a composite, multi-value attribute

- [ ] True
- [ ] False

### Question 33
In SQL, an equal join is a kind of natural join

- [ ] True
- [ ] False

### Question 34
Determine which of the following SQL query/queries always produce(s) the same results as SELECT A FROM (SELECT * FROM B WHERE C = 1) WHERE D = 2

- [ ] SELECT* FROM (SELECT A FROM B WHERE C = 1) WHERE D = 2
- [ ] None of the others
- [ ] SELECT A FROM (SELECT * FROM B WHERE (C = 1 OR D = 2))
- [ ] SELECT DISTINCT A FROM (SELECT* FROM B WHERE C = 1) WHERE D = 2
- [ ] SELECT A FROM B WHERE (C = 1 AND D = 2)

### Question 35
For the fast version of the Bubble sort with the swapped flag, it

- [ ] None of the others
- [ ] always terminates early by the swapped flag
- [ ] always swaps non-adjacent items
- [ ] has the same best-case time complexity as that of insertion sort
- [ ] has the same worst-case time complexity as that of selection sort

### Question 36
Given a relation schema R = (A, B, C, D) with the set X of functional dependencies: A-> B, A->C, B->D, which of the
following functional dependency(ies) is/are in $X^+$.

- [ ] C->D
- [ ] None of the others
- [ ] A->D
- [ ] AC->CD
- [ ] CD->D

### Question 37
The cascade rule deletes the corresponding rows in both the parent table and the dependent table

- [ ] True
- [ ] False

### Question 38
An associate entity can
- [ ] have attributes
- [ ] be a relationship
- [ ] None of the others
- [ ] be an entity
- [ ] have no attributes

### Question 39
Relational algebra cannot simulate
- [ ] the natural join operation
- [ ] the inner join operation
- [ ] the vector aggregation function in SQL
- [ ] None of the others
- [ ] the outer join operation

### Question 40
The average-case time complexity of binary search is the same as its worst-case time complexity.

- [ ] True
- [ ] False

### Question 41
### Question 42