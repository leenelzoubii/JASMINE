$workDir = "C:\Users\HP\autism-screening-pose\jasmine-next"
$logFile = Join-Path $workDir ".server.log"
$process = Start-Process -NoNewWindow -FilePath "npx" -ArgumentList "next dev -p 3000" -WorkingDirectory $workDir -RedirectStandardOutput $logFile -RedirectStandardError ($logFile + ".err") -PassThru
Write-Output "Server PID: $($process.Id)"
