/**
 * Settings-redirect migration: prose that names a Settings tab now renders the
 * tab name as a <SettingsLink> (via the react-i18next <Trans> `<0>` idiom)
 * instead of a hand-written `/settings/...` string or plain text.
 *
 * Mounts the lightest migrated component and asserts the anchor href. The
 * heavier host (InstancesViewport) is covered through its catalog value + the
 * shared idiom, rendered standalone below, since mounting it needs the full
 * api / query / store scaffolding.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Trans } from 'react-i18next'
import type { ReactNode } from 'react'
import { i18nT } from '../i18n/t'
import { SettingsLink } from '../components/SettingsLink'
import VoiceDisabledModal from '../components/VoiceDisabledModal'

const wrap = (ui: ReactNode) => render(<MemoryRouter>{ui}</MemoryRouter>)

describe('Settings-link migrations', () => {
  it('VoiceDisabledModal: "Settings → Voice" is a link to /settings/voice', () => {
    const onOpenSettings = vi.fn()
    wrap(<VoiceDisabledModal open onClose={() => {}} onOpenSettings={onOpenSettings} />)
    const link = screen.getByRole('link', {
      name: i18nT('components.voiceDisabledModal.settings_voice'),
    }) as HTMLAnchorElement
    expect(link.getAttribute('href')).toBe('/settings/voice')
    // A plain click routes through the host's embed-aware handler (the same
    // one the footer button uses) instead of the bare href.
    fireEvent.click(link)
    expect(onOpenSettings).toHaveBeenCalledTimes(1)
  })

  it('VoiceDisabledModal (unavailable): "Settings → Voice" links too, via the catalog <0> idiom', () => {
    const onOpenSettings = vi.fn()
    wrap(<VoiceDisabledModal open reason="unavailable" provider="whisper" onClose={() => {}} onOpenSettings={onOpenSettings} />)
    const link = screen.getByRole('link', { name: /Settings → Voice/ }) as HTMLAnchorElement
    expect(link.getAttribute('href')).toBe('/settings/voice')
    fireEvent.click(link)
    expect(onOpenSettings).toHaveBeenCalledTimes(1)
  })

  it('InstancesViewport hint: catalog value + idiom yield a link to /settings/instances', () => {
    wrap(
      <Trans
        i18nKey="components.instancesViewport.this_tab_stays_until_you_disconnect_the_instance"
        components={[<SettingsLink key="l" tab="instances" />]}
      />,
    )
    const link = screen.getByRole('link', { name: /Settings → Remote Instances/ }) as HTMLAnchorElement
    expect(link.getAttribute('href')).toBe('/settings/instances')
    // The sentence around the link survives the wrapping.
    expect(screen.getByText(/This tab stays until you disconnect the instance in/)).toBeInTheDocument()
  })
})
