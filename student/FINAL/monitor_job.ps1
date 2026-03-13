# Made with the help of Claude Opus

param(
    [Parameter(Mandatory=$true)]
    [string]$JobName,

    [int]$IntervalSeconds = 30
)

Write-Host "Monitoring SageMaker job: $JobName" -ForegroundColor Cyan
Write-Host "Checking every ${IntervalSeconds}s... (Ctrl+C to stop)" -ForegroundColor Gray
Write-Host ""

while ($true) {
    try {
        $job = aws sagemaker describe-training-job --training-job-name $JobName --output json | ConvertFrom-Json
        $status = $job.TrainingJobStatus
        $secondary = $job.SecondaryStatus
        $elapsed = ""

        if ($job.TrainingStartTime) {
            $start = [DateTime]::Parse($job.TrainingStartTime)
            $duration = (Get-Date) - $start
            $elapsed = " | Elapsed: {0:hh\:mm\:ss}" -f $duration
        }

        $color = switch ($status) {
            "InProgress" { "Yellow" }
            "Completed"  { "Green" }
            "Failed"     { "Red" }
            "Stopped"    { "Gray" }
            default      { "White" }
        }

        Write-Host "$(Get-Date -Format 'HH:mm:ss') - $status ($secondary)$elapsed" -ForegroundColor $color

        if ($status -ne "InProgress") {
            Write-Host ""

            if ($status -eq "Completed") {
                Write-Host "Job completed successfully!" -ForegroundColor Green
                Write-Host "Download with:" -ForegroundColor Gray
                Write-Host "  python download_model.py --job_name $JobName" -ForegroundColor White
            }
            elseif ($status -eq "Failed") {
                Write-Host "Job failed!" -ForegroundColor Red
                Write-Host "Reason: $($job.FailureReason)" -ForegroundColor Red
            }

            # Beeps first, then popup (popup blocks until closed)
            [console]::beep(1000, 500)
            [console]::beep(1000, 500)
            [console]::beep(1000, 500)

            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.MessageBox]::Show(
                "SageMaker job finished: $status`n`nJob: $JobName",
                "TinyStories Training",
                "OK",
                $(if ($status -eq "Completed") { "Information" } else { "Error" })
            ) | Out-Null

            break
        }
    }
    catch {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Error polling: $_" -ForegroundColor Red
    }

    Start-Sleep -Seconds $IntervalSeconds
}