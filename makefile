CC=gcc
CFLAGS= -Wall -Wextra --pedantic-errors -O3 -fopenmp

ifeq ($(OS),Windows_NT) 
RM = del
else
RM= rm
endif

nn: main.o matrix.o nn.o activation.o loss.o optimizer.o eval.o
	$(CC) $(CFLAGS) -o $@ $^

# matrix: matrix.o
# 	$(CC) $(CFLAGS) -o $@ $^

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c  $<

loss.o: loss.c loss.h matrix.h
	$(CC) $(CFLAGS) -c $<

activation.o: activation.c activation.h
	$(CC) $(CFLAGS) -c $<

optimizer.o: optimizer.c optimizer.h
	$(CC) $(CFLAGS) -c $<

nn.o: nn.c matrix.h nn.h loss.h optimizer.h
	$(CC) $(CFLAGS) -c $<

eval.o: eval.c eval.h
	$(CC) $(CFLAGS) -c $<

main.o: main.c nn.h matrix.h loss.h activation.h
	$(CC) $(CFLAGS) -c $<


clean: 
	$(RM) *.o
	$(RM) *.exe
	$(RM) *.gch
