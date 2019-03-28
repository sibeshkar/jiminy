upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*

test:
	find . -name '*.pyc' -delete
	docker build -f test.dockerfile -t quay.io/boxware/jiminy:test .
	docker run -v /usr/bin/docker:/usr/bin/docker -v /root/.docker:/root/.docker -v /var/run/docker.sock:/var/run/docker.sock --net=host quay.io/boxware/jiminy:test

build:
	find . -name '*.pyc' -delete
	docker build -t quay.io/boxware/jiminy .
	docker build -f test.dockerfile -t quay.io/boxware/jiminy:test .

push:
	find . -name '*.pyc' -delete
	docker build -t quay.io/boxware/jiminy .
	docker build -f test.dockerfile -t quay.io/boxware/jiminy:test .

	docker push quay.io/boxware/jiminy
	docker push quay.io/boxware/jiminy:test

test-push:
	docker build -f test.dockerfile -t quay.io/boxware/jiminy:test .
	docker push quay.io/boxware/jiminy:test
