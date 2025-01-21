PROGS = sample_xor sample_mnist sample_graph
LIB = fnn
CC = g++
FLAGS = -Wall -Wextra -O4

all: $(PROGS)

# Make sample programs
$(PROGS): %: %.o $(LIB).o
	$(CC) $< $(LIB).o -o $@ $(FLAGS)

%.o: %.cpp
	$(CC) -c $< -o $@ $(FLAGS)

# Make library (FNN)
$(LIB).o: $(LIB).cpp
	$(CC) -c $(LIB).cpp $(FLAGS)

# remove object and executable files
clean:
	rm -f *.o $(PROGS)

# remove all .model files
clean-model:
	rm -f *.model