import scrapy
from stat_plate.items import StatPlateItem


class PlateSpider(scrapy.Spider):
    name = 'plate'
    url_template = 'http://tjj.yancheng.gov.cn/SJFB/YDSJ/index_{0}.html'

    def start_requests(self):
        urls = [
            'http://tjj.yancheng.gov.cn/SJFB/YDSJ/index.html',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for item in response.xpath("//a[re:test(text(), '^统计月报')]"):
            href = item.css('a::attr(href)').extract_first()
            yield scrapy.Request(response.urljoin(href), callback=self.parse_download_page)
        for i in range(1, 7):
            next_page = self.url_template.format(i)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_download_page(self, response):
        selector = response.xpath("//a[re:test(text(), '月报')]")
        item = StatPlateItem()
        item['url'] = response.urljoin(selector.css('a::attr(href)').extract_first())
        item['name'] = selector.css('a::text').extract_first()
        yield item
