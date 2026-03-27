import blpapi

opts = blpapi.SessionOptions()
opts.setServerHost('localhost')
opts.setServerPort(8194)
s = blpapi.Session(opts)
s.start()
s.openService('//blp/exrsvc')
svc = s.getService('//blp/exrsvc')

req = svc.createRequest('ExcelGetGridRequest')
print(req.asElement())
s.stop()