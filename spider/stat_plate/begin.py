from scrapy import cmdline

cmdline.execute("scrapy crawl plate".split())
# cmdline.execute("scrapy crawl plate -s JOBDIR=crawls/somespider-1".split())