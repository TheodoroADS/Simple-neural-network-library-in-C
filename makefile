CC=gcc
CFLAGS= -Wall -Wextra --pedantic-errors -O3

nn: main.o matrix.o nn.o activation.o loss.o
	$(CC) $(CFLAGS) -o $@ $^

# matrix: matrix.o
# 	$(CC) $(CFLAGS) -o $@ $^

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c  $<

loss.o: loss.c loss.h matrix.h
	$(CC) $(CFLAGS) -c $<

activation.o: activation.c activation.h
	$(CC) $(CFLAGS) -c $<

nn.o: nn.c matrix.h nn.h loss.h
	$(CC) $(CFLAGS) -c $<

main.o: main.c nn.h matrix.h loss.h activation.h
	$(CC) $(CFLAGS) -c $<
