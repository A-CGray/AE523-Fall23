default:
	jupyter-book build .

.phony: clean
clean:
	rm -rf _build
