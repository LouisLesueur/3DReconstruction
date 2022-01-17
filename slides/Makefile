SOURCES = \
	slides \

BUILDDIR = build/
DEP = $(wildcard *.sty *.tex *.jpg *.png)
TARGETS = $(addprefix $(BUILDDIR),$(addsuffix .pdf,$(SOURCES)))

PARAMETERS += --pdf-engine-opt=-shell-escape --citeproc

all: $(TARGETS)

$(BUILDDIR)slides.pdf : slides.md $(DEP)
	@mkdir -p $(BUILDDIR) # Make sure build dir exists
	pandoc $(PARAMETERS) -t beamer $< -o $@

clean:
	@rm -f $(TARGETS)
