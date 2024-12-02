PROG = sample
LIB = fnn
CC = g++
FLAGS = -Wall -Wextra -g

all: $(PROG)

$(PROG): $(PROG).o $(LIB).o
	$(CC) $(PROG).o $(LIB).o $(FLAGS) -o $(PROG)

$(PROG).o: $(PROG).cpp
	$(CC) -c $(PROG).cpp $(FLAGS)

$(LIB).o: $(LIB).cpp
	$(CC) -c $(LIB).cpp $(FLAGS)

clean:
	rm *.o $(PROG)