PROG = sampleProg
CC = g++
FLAGS = -Wall -Wextra -g

all: $(PROG)

$(PROG): $(PROG).o mdp.o mdpValueIterator.o
	$(CC) $(PROG).o mdp.o mdpValueIterator.o $(FLAGS) -o $(PROG)

$(PROG).o: $(PROG).cpp
	$(CC) -c $(PROG).cpp $(FLAGS)

mdp.o: mdp.cpp
	$(CC) -c mdp.cpp $(FLAGS)

mdpValueIterator.o: mdpValueIterator.cpp
	$(CC) -c mdpValueIterator.cpp $(FLAGS)

clean:
	rm *.o $(PROG)