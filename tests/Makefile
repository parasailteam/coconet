all: fusion-test storeat-test
	./fusion-test
	./storeat-test

fusion-test: fusion-test.cpp
	g++ $< ../src/codegen.cpp ../src/dsl.cpp ../src/pipeline.cpp -I../googletest/googletest/include/ -I../src/ -L../googletest/build/lib -lgtest -lgtest_main -lpthread -g -O0 -DNDEBUG -o $@ 

storeat-test: storeat-test.cpp
	g++ $< ../src/codegen.cpp ../src/dsl.cpp ../src/pipeline.cpp -I../googletest/googletest/include/ -I../src/ -L../googletest/build/lib -lgtest -lgtest_main -lpthread -g -O0 -DNDEBUG -o $@ 


clean:
	rm fusion-test
	rm storeat-test