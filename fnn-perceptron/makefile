PROG = sample
PROG2 = sample_mnist
LIB = fnn
CC = g++
FLAGS = -Wall -Wextra -g

all: $(PROG) $(PROG2)

# Make sample program 1 (xor)
$(PROG): $(PROG).o $(LIB).o
	$(CC) $(PROG).o $(LIB).o $(FLAGS) -o $(PROG)

$(PROG).o: $(PROG).cpp
	$(CC) -c $(PROG).cpp $(FLAGS)

# Make sample program 2 (mnist)
$(PROG2): $(PROG2).o $(LIB).o
	$(CC) $(PROG2).o $(LIB).o $(FLAGS) -o $(PROG2)

$(PROG2).o: $(PROG2).cpp
	$(CC) -c $(PROG2).cpp $(FLAGS)

# Make library (FNN)
$(LIB).o: $(LIB).cpp
	$(CC) -c $(LIB).cpp $(FLAGS)

clean:
	rm *.o $(PROG) $(PROG2)

clean-model:
	rm *.model